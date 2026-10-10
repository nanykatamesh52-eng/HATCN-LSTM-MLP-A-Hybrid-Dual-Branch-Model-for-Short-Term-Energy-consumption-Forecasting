#!/usr/bin/env python3
import math, copy, random, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")
CSV_PATH="PJME_hourly.csv"
OUT_DIR=Path("informer_probsparse_results")
LOOKBACK,HORIZON,LABEL_LEN=168,24,24
QUANTILES=[0.1,0.5,0.9]
SEEDS=[0,1,2]
EPOCHS,PATIENCE,BATCH_SIZE=60,10,64
LR,WEIGHT_DECAY=1e-3,1e-5
D_MODEL,N_HEADS,E_LAYERS,D_LAYERS,D_FF,DROPOUT=128,8,2,1,256,0.1
FACTOR=5
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def holiday_flag(idx):
    import calendar as pycalendar
    out=set()
    for y in sorted(set(idx.year)):
        fixed=[pd.Timestamp(y,1,1),pd.Timestamp(y,6,19),pd.Timestamp(y,7,4),
               pd.Timestamp(y,11,11),pd.Timestamp(y,12,25)]
        for d in fixed:
            out.add(d.normalize())
            if d.weekday()==5: out.add((d-pd.Timedelta(days=1)).normalize())
            if d.weekday()==6: out.add((d+pd.Timedelta(days=1)).normalize())

        def month_days(month):
            last=pycalendar.monthrange(int(y), int(month))[1]
            return pd.date_range(
                f"{int(y)}-{int(month):02d}-01",
                f"{int(y)}-{int(month):02d}-{last:02d}",
                freq="D"
            )

        def nth(month,weekday,n):
            r=month_days(month)
            r=r[r.weekday==weekday]
            return r[n-1]

        out.add(nth(1,0,3))   # MLK
        out.add(nth(2,0,3))   # Washington's Birthday

        r=month_days(5)
        out.add(r[r.weekday==0][-1])  # Memorial Day

        out.add(nth(9,0,1))   # Labor Day
        out.add(nth(10,0,2))  # Columbus Day
        out.add(nth(11,3,4))  # Thanksgiving

    return np.array(
        [float(x.normalize() in out) for x in idx],
        dtype=np.float32
    )

def calendar(idx):
    idx=pd.DatetimeIndex(idx)
    h=idx.hour.to_numpy(float); w=idx.dayofweek.to_numpy(float); m=idx.month.to_numpy(float)
    return np.column_stack([
        np.sin(2*np.pi*h/24),np.cos(2*np.pi*h/24),
        np.sin(2*np.pi*w/7),np.cos(2*np.pi*w/7),
        np.sin(2*np.pi*(m-1)/12),np.cos(2*np.pi*(m-1)/12),
        (idx.dayofweek>=5).astype(float),holiday_flag(idx)
    ]).astype(np.float32)

def load():
    df=pd.read_csv(CSV_PATH)
    df["Datetime"]=pd.to_datetime(df["Datetime"])
    col="PJME_MW" if "PJME_MW" in df.columns else ("MW" if "MW" in df.columns else [c for c in df.columns if c!="Datetime"][0])
    df=df.sort_values("Datetime").drop_duplicates("Datetime")[["Datetime",col]].dropna()
    return df[col].to_numpy(np.float32),pd.DatetimeIndex(df.Datetime)

def windows(v,dt):
    E=calendar(dt); X=[]; EE=[]; ED=[]; Y=[]
    for t in range(LOOKBACK,len(v)-HORIZON+1):
        X.append(v[t-LOOKBACK:t]); EE.append(E[t-LOOKBACK:t])
        dec_load=np.r_[v[t-LABEL_LEN:t],np.zeros(HORIZON,np.float32)]
        ED.append(np.column_stack([dec_load,E[t-LABEL_LEN:t+HORIZON]]))
        Y.append(v[t:t+HORIZON])
    return map(lambda z:np.asarray(z,np.float32),(X,EE,ED,Y))

class DS(Dataset):
    def __init__(self,x,e,d,y):
        self.x=torch.from_numpy(x); self.e=torch.from_numpy(e); self.d=torch.from_numpy(d); self.y=torch.from_numpy(y)
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.x[i],self.e[i],self.d[i],self.y[i]

class Scale:
    def fit(self,x): self.a=float(x.min()); self.b=float(x.max()); self.s=self.b-self.a
    def tr(self,x): return (x-self.a)/self.s
    def inv(self,x): return x*self.s+self.a

class PE(nn.Module):
    def __init__(self,d,n=512):
        super().__init__(); p=torch.zeros(n,d); pos=torch.arange(n).float().unsqueeze(1)
        div=torch.exp(torch.arange(0,d,2).float()*(-math.log(10000)/d))
        p[:,0::2]=torch.sin(pos*div); p[:,1::2]=torch.cos(pos*div)
        self.register_buffer("p",p.unsqueeze(0))
    def forward(self,x): return x+self.p[:,:x.size(1)]

class ProbSparse(nn.Module):
    def __init__(self,d,h,factor=5,drop=.1):
        super().__init__(); assert d%h==0
        self.d=d; self.h=h; self.dk=d//h; self.factor=factor
        self.q=nn.Linear(d,d); self.k=nn.Linear(d,d); self.v=nn.Linear(d,d); self.o=nn.Linear(d,d); self.drop=nn.Dropout(drop)
    def forward(self,q,k=None,v=None,causal=False):
        if k is None:k=q
        if v is None:v=k
        B,Lq,_=q.shape; Lk=k.shape[1]; D=self.dk
        q=self.q(q).view(B,Lq,self.h,D).transpose(1,2)
        k=self.k(k).view(B,Lk,self.h,D).transpose(1,2)
        v=self.v(v).view(B,Lk,self.h,D).transpose(1,2)
        sk=min(Lk,max(1,int(self.factor*math.ceil(math.log(Lk+1)))))
        st=min(Lq,max(1,int(self.factor*math.ceil(math.log(Lq+1)))))
        ids=torch.randint(Lk,(sk,),device=q.device)
        score_s=torch.matmul(q,k[:,:,ids,:].transpose(-2,-1))/math.sqrt(D)
        M=score_s.max(-1).values-score_s.mean(-1)
        top=torch.topk(M,st,dim=-1).indices
        if causal:
            den=torch.arange(1,Lk+1,device=v.device,dtype=v.dtype).view(1,1,Lk,1)
            ctx=v.cumsum(2)/den
        else:
            ctx=v.mean(2,keepdim=True).expand(B,self.h,Lq,D).clone()
        qt=torch.gather(q,2,top.unsqueeze(-1).expand(-1,-1,-1,D))
        sc=torch.matmul(qt,k.transpose(-2,-1))/math.sqrt(D)
        if causal:
            kp=torch.arange(Lk,device=q.device).view(1,1,1,Lk)
            sc=sc.masked_fill(kp>top.unsqueeze(-1),float("-inf"))
        a=self.drop(torch.softmax(sc,-1)); z=torch.matmul(a,v)
        ctx.scatter_(2,top.unsqueeze(-1).expand(-1,-1,-1,D),z)
        return self.o(ctx.transpose(1,2).contiguous().view(B,Lq,self.d))

class FullAttn(nn.Module):
    def __init__(self,d,h,drop=.1):
        super().__init__(); assert d%h==0
        self.d=d;self.h=h;self.dk=d//h
        self.q=nn.Linear(d,d);self.k=nn.Linear(d,d);self.v=nn.Linear(d,d);self.o=nn.Linear(d,d);self.drop=nn.Dropout(drop)
    def forward(self,q,k,v):
        B,Lq,_=q.shape;Lk=k.shape[1]
        q=self.q(q).view(B,Lq,self.h,self.dk).transpose(1,2)
        k=self.k(k).view(B,Lk,self.h,self.dk).transpose(1,2)
        v=self.v(v).view(B,Lk,self.h,self.dk).transpose(1,2)
        a=self.drop(torch.softmax(torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.dk),-1))
        z=torch.matmul(a,v).transpose(1,2).contiguous().view(B,Lq,self.d)
        return self.o(z)

class FF(nn.Module):
    def __init__(self,d,ff,drop):
        super().__init__();self.m=nn.Sequential(nn.Linear(d,ff),nn.GELU(),nn.Dropout(drop),nn.Linear(ff,d),nn.Dropout(drop))
    def forward(self,x):return self.m(x)

class EncLayer(nn.Module):
    def __init__(self,d,h,ff,drop):
        super().__init__();self.a=ProbSparse(d,h,FACTOR,drop);self.f=FF(d,ff,drop);self.n1=nn.LayerNorm(d);self.n2=nn.LayerNorm(d)
    def forward(self,x):
        z = self.n1(x + self.a(x))
        return self.n2(z + self.f(z))

class DecLayer(nn.Module):
    def __init__(self,d,h,ff,drop):
        super().__init__();self.a=ProbSparse(d,h,FACTOR,drop);self.c=FullAttn(d,h,drop);self.f=FF(d,ff,drop)
        self.n1=nn.LayerNorm(d);self.n2=nn.LayerNorm(d);self.n3=nn.LayerNorm(d)
    def forward(self,x,m):
        x=self.n1(x+self.a(x,causal=True));x=self.n2(x+self.c(x,m,m));return self.n3(x+self.f(x))

class QHead(nn.Module):
    def __init__(self,d):
        super().__init__();self.b=nn.Linear(d,1);self.d1=nn.Linear(d,1);self.d2=nn.Linear(d,1)
    def forward(self,x):
        q1=self.b(x);q2=q1+F.softplus(self.d1(x));q3=q2+F.softplus(self.d2(x));return torch.cat([q1,q2,q3],-1)

class Informer(nn.Module):
    def __init__(self):
        super().__init__()
        self.ee=nn.Linear(9,D_MODEL);self.de=nn.Linear(9,D_MODEL);self.pe=PE(D_MODEL)
        self.enc=nn.ModuleList([EncLayer(D_MODEL,N_HEADS,D_FF,DROPOUT) for _ in range(E_LAYERS)])
        self.dec=nn.ModuleList([DecLayer(D_MODEL,N_HEADS,D_FF,DROPOUT) for _ in range(D_LAYERS)])
        self.ne=nn.LayerNorm(D_MODEL);self.nd=nn.LayerNorm(D_MODEL);self.head=QHead(D_MODEL)
    def forward(self,x,e,d):
        z=self.pe(self.ee(torch.cat([x.unsqueeze(-1),e],-1)))
        for l in self.enc:z=l(z)
        z=self.ne(z)
        y=self.pe(self.de(d))
        for l in self.dec:y=l(y,z)
        return self.head(self.nd(y[:,-HORIZON:]))

def loss(q,y):
    return torch.stack([torch.maximum(qq*(y-q[...,i]),(qq-1)*(y-q[...,i])).mean() for i,qq in enumerate(QUANTILES)]).mean()

def epoch(model,loader,opt=None):
    train=opt is not None;model.train(train);tot=n=0
    for x,e,d,y in loader:
        x,e,d,y=x.to(DEVICE),e.to(DEVICE),d.to(DEVICE),y.to(DEVICE)
        if train:opt.zero_grad(set_to_none=True)
        L=loss(model(x,e,d),y)
        if train:L.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        tot+=L.item()*len(y);n+=len(y)
    return tot/n

@torch.no_grad()
def pred(model,loader):
    model.eval();a=[];b=[]
    for x,e,d,y in loader:a.append(model(x.to(DEVICE),e.to(DEVICE),d.to(DEVICE)).cpu().numpy());b.append(y.numpy())
    return np.concatenate(a),np.concatenate(b)

def metrics(q,y,s):
    q1,q2,q3=q[...,0],q[...,1],q[...,2];err=q2-y
    cov=np.mean((y>=q1)&(y<=q3));w=q3-q1;alpha=.2
    wi=w.copy();lo=y<q1;hi=y>q3
    wi[lo]+=(2/alpha)*(q1[lo]-y[lo]);wi[hi]+=(2/alpha)*(y[hi]-q3[hi])
    pb=[np.mean(np.maximum(qq*(y-q[...,i]),(qq-1)*(y-q[...,i]))) for i,qq in enumerate(QUANTILES)]
    em=s.inv(err);ym=s.inv(y);q1m=s.inv(q1);q3m=s.inv(q3)
    return dict(PICP=float(cov),coverage_gap=float(cov-.8),PINAW=float(w.mean()),Winkler=float(wi.mean()),
                pinball_q10=float(pb[0]),pinball_q50=float(pb[1]),pinball_q90=float(pb[2]),pinball_mean=float(np.mean(pb)),
                crossing_rate=float(np.mean((q1>q2)|(q2>q3))),RMSE=float(np.sqrt(np.mean(err**2))),MAE=float(np.mean(abs(err))),
                MSE=float(np.mean(err**2)),MAPE=float(np.mean(abs(em)/np.maximum(abs(ym),1))*100),
                RMSE_original=float(np.sqrt(np.mean((q2m:=s.inv(q2)-ym)**2))),MAE_original=float(np.mean(abs(q2m))),
                MSE_original=float(np.mean(q2m**2)),MAPE_original=float(np.mean(abs(q2m)/np.maximum(abs(ym),1))*100))

def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Using device:",DEVICE)
    v,dt=load();print("Dataset:");print("records =",len(v));print("start  =",dt[0]);print("end    =",dt[-1])
    X,E,D,Y=windows(v,dt);X,E,D,Y=list(X),list(E),list(D),list(Y)
    n=len(Y);nt=int(.70*n);nv=int(.15*n)
    parts=[(X[:nt],E[:nt],D[:nt],Y[:nt]),(X[nt:nt+nv],E[nt:nt+nv],D[nt:nt+nv],Y[nt:nt+nv]),(X[nt+nv:],E[nt+nv:],D[nt+nv:],Y[nt+nv:])]
    for i,p in enumerate(parts):print(f"Split {i}: X={np.asarray(p[0]).shape}, E={np.asarray(p[1]).shape}, D={np.asarray(p[2]).shape}, Y={np.asarray(p[3]).shape}")
    sc=Scale();sc.fit(np.asarray(parts[0][3]));print("Training load range:");print(f"min = {sc.a:.4f}");print(f"max = {sc.b:.4f}")
    norm=[]
    for x,e,d,y in parts:
        x=np.asarray(x,np.float32);e=np.asarray(e,np.float32);d=np.asarray(d,np.float32);y=np.asarray(y,np.float32)
        x=sc.tr(x).astype(np.float32);y=sc.tr(y).astype(np.float32);d=d.copy();d[:,:,0]=sc.tr(d[:,:,0])
        norm.append((x,e,d,y))
    loaders=[]
    for i,(x,e,d,y) in enumerate(norm):
        loaders.append(DataLoader(DS(x,e,d,y),batch_size=BATCH_SIZE,shuffle=(i==0),pin_memory=DEVICE.type=="cuda"))
    rows=[]
    for seed in SEEDS:
        print("\n"+"="*70);print("STARTING INFORMER-PROBSPARSE SEED",seed);print("="*70);seed_all(seed)
        m=Informer().to(DEVICE);opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
        best=float("inf");state=None;bad=0;hist=[]
        for ep in range(1,EPOCHS+1):
            tr=epoch(m,loaders[0],opt);va=epoch(m,loaders[1])
            hist.append([ep,tr,va]);print(f"seed {seed} | epoch {ep:02d} | train {tr:.5f} | val {va:.5f}")
            if va<best-1e-6:best=va;state=copy.deepcopy(m.state_dict());bad=0
            else:bad+=1
            if bad>=PATIENCE:print("Early stopping at epoch",ep);break
        m.load_state_dict(state);q,y=pred(m,loaders[2]);r=metrics(q,y,sc);r.update(seed=seed,best_val_loss=best,epochs_ran=len(hist));rows.append(r)
        pd.DataFrame(hist,columns=["epoch","train_loss","val_loss"]).to_csv(OUT_DIR/f"informer_probsparse_loss_history_seed_{seed}.csv",index=False)
        np.save(OUT_DIR/f"informer_probsparse_predictions_seed_{seed}.npy",q)
    df=pd.DataFrame(rows);df.to_csv(OUT_DIR/"informer_probsparse_seed_results.csv",index=False)
    print("\n"+"="*70);print("FINAL INFORMER-PROBSPARSE NORMALIZED RESULTS");print("="*70)
    for c in ["PICP","coverage_gap","PINAW","Winkler","pinball_q10","pinball_q50","pinball_q90","pinball_mean","crossing_rate","RMSE","MAE","MSE","MAPE"]:
        print(f"{c:25s} {df[c].mean():.4f} +- {df[c].std(ddof=1):.4f}")
    print("\nOriginal MW: RMSE %.4f +- %.4f | MAE %.4f +- %.4f | MAPE %.4f +- %.4f" %
          (df.RMSE_original.mean(),df.RMSE_original.std(ddof=1),df.MAE_original.mean(),df.MAE_original.std(ddof=1),df.MAPE_original.mean(),df.MAPE_original.std(ddof=1)))
    print("\nSaved:",OUT_DIR)

if __name__=="__main__":main()
