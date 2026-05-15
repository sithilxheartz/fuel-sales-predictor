"""
SALES MODEL EVALUATION SCRIPT — EMERALD LANKA
==============================================
Generates comprehensive evaluation charts for the research report.

Produces 6 figures:
  sf1_overview.png        — Dataset overview & sales trends
  sf2_accuracy.png        — Actual vs predicted (all 4 fuels, 2x4 grid)
  sf3_cv_validation.png   — Cross-validation results & residuals
  sf4_features.png        — Feature importance per fuel
  sf5_seasonal.png        — Seasonal & temporal patterns
  sf6_summary.png         — Performance summary table

Run:
  python scripts/evaluate_sales_model.py

Output saved to: evaluation_output/ folder
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = "evaluation_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── COLORS ──────────────────────────────────────────────────────────────────
BG='#0B0D0C'; SURFACE='#161B19'; GREEN='#00FF88'; DKGREEN='#00A35C'
WHITE='#FFFFFF'; DIM='#9E9E9E'; ERROR='#FF5252'; WARNING='#FFAB40'
BLUE='#40C4FF'; PURPLE='#CE93D8'

FUEL_COLORS = {'petrol':BLUE,'super_petrol':GREEN,'diesel':WARNING,'super_diesel':PURPLE}
FUEL_LABELS = {'petrol':'92 Petrol','super_petrol':'95 Petrol',
               'diesel':'Auto Diesel','super_diesel':'Super Diesel'}
FUELS       = ['petrol','super_petrol','diesel','super_diesel']
MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

XGBOOST_PARAMS = {
    'petrol':       {'n_estimators':300,'max_depth':4,'learning_rate':0.05},
    'super_petrol': {'n_estimators':300,'max_depth':3,'learning_rate':0.05},
    'diesel':       {'n_estimators':300,'max_depth':4,'learning_rate':0.05},
    'super_diesel': {'n_estimators':300,'max_depth':3,'learning_rate':0.05},
}
BASE_FEATURES = ['month','day_of_week','day_of_year','quarter',
                 'month_sin','month_cos','is_monsoon','is_dry_season','is_weekend']

plt.rcParams.update({
    'figure.facecolor':BG,'axes.facecolor':SURFACE,'axes.edgecolor':'#2A3530',
    'axes.labelcolor':DIM,'xtick.color':DIM,'ytick.color':DIM,'text.color':WHITE,
    'grid.color':'#1F2820','grid.linewidth':0.8,'font.family':'DejaVu Sans',
    'font.size':10,'axes.titlesize':11,'axes.titlecolor':WHITE,
    'axes.titleweight':'bold','legend.facecolor':SURFACE,
    'legend.edgecolor':'#2A3530','legend.labelcolor':WHITE,
})

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def sax(ax, title='', xl='', yl=''):
    ax.set_facecolor(SURFACE)
    ax.spines[['top','right','left','bottom']].set_color('#2A3530')
    if title: ax.set_title(title,color=WHITE,fontsize=11,fontweight='bold',pad=10)
    if xl: ax.set_xlabel(xl,color=DIM,fontsize=9)
    if yl: ax.set_ylabel(yl,color=DIM,fontsize=9)
    ax.grid(True,alpha=0.3,linestyle='--')

def calc_mape(a,p):
    a,p=np.array(a),np.array(p); m=a>0
    return np.mean(np.abs((a[m]-p[m])/a[m]))*100

def save(fig,name):
    path=os.path.join(OUTPUT_DIR,name)
    fig.savefig(path,dpi=150,bbox_inches='tight',facecolor=BG,edgecolor='none')
    plt.close(fig); print(f"  ✅ Saved: {path}")

# ─── DATA ────────────────────────────────────────────────────────────────────

def load_data(csv_path="data/sales_data_clean.csv"):
    paths=[csv_path,"data/sales_data.csv","data/fuel_sales_evaporation.csv"]
    df=None
    for p in paths:
        if os.path.exists(p):
            df=pd.read_csv(p); df['date']=pd.to_datetime(df['date'])
            print(f"  Loaded: {p} ({len(df)} rows)"); break
    if df is None: raise FileNotFoundError("No sales data found")

    rename={'petrol_sales':'petrol','super_petrol_sales':'super_petrol',
            'diesel_sales':'diesel','super_diesel_sales':'super_diesel',
            'petrol_sales_L':'petrol','super_petrol_sales_L':'super_petrol',
            'diesel_sales_L':'diesel','super_diesel_sales_L':'super_diesel'}
    df=df.rename(columns=rename)
    for f in FUELS:
        if f not in df.columns: df[f]=0.0

    df['month']=df['date'].dt.month; df['day_of_week']=df['date'].dt.dayofweek
    df['day_of_year']=df['date'].dt.dayofyear; df['quarter']=df['date'].dt.quarter
    df['month_sin']=np.sin(2*np.pi*df['month']/12)
    df['month_cos']=np.cos(2*np.pi*df['month']/12)
    df['is_monsoon']=df['month'].isin([5,6,7,8,9]).astype(int)
    df['is_dry_season']=df['month'].isin([12,1,2,3,4]).astype(int)
    df['is_weekend']=df['day_of_week'].isin([5,6]).astype(int)

    for fuel in FUELS:
        df[f'{fuel}_lag1']=df[fuel].shift(1); df[f'{fuel}_lag7']=df[fuel].shift(7)
        df[f'{fuel}_roll7']=df[fuel].shift(1).rolling(7,min_periods=1).mean()
        df[f'{fuel}_roll30']=df[fuel].shift(1).rolling(30,min_periods=1).mean()

    df=df.dropna().sort_values('date').reset_index(drop=True)
    print(f"  Ready: {len(df)} rows | {df['date'].min().date()} → {df['date'].max().date()}")
    return df

# ─── EVALUATION ──────────────────────────────────────────────────────────────

def evaluate_all(df):
    results={}
    for fuel in FUELS:
        print(f"  [{fuel}] evaluating...")
        features=BASE_FEATURES+[f'{fuel}_lag1',f'{fuel}_lag7',
                                  f'{fuel}_roll7',f'{fuel}_roll30']
        mask=df[fuel]>0; X=df[features][mask]; y=df[fuel][mask]
        dates=df['date'][mask]

        tscv=TimeSeriesSplit(n_splits=5)
        cv_mapes,cv_maes,fp,fa,fd=[],[],[],[],[]
        for ti,vi in tscv.split(X):
            m=XGBRegressor(subsample=0.8,colsample_bytree=0.8,
                           random_state=42,verbosity=0,**XGBOOST_PARAMS[fuel])
            m.fit(X.iloc[ti],y.iloc[ti])
            p=np.maximum(0,m.predict(X.iloc[vi]))
            cv_maes.append(mean_absolute_error(y.iloc[vi],p))
            cv_mapes.append(calc_mape(y.iloc[vi].values,p))
            fp.extend(p.tolist()); fa.extend(y.iloc[vi].tolist())
            fd.extend(dates.iloc[vi].tolist())

        X_tr,X_te=X.iloc[:-20],X.iloc[-20:]
        y_tr,y_te=y.iloc[:-20],y.iloc[-20:]
        hm=XGBRegressor(subsample=0.8,colsample_bytree=0.8,
                        random_state=42,verbosity=0,**XGBOOST_PARAMS[fuel])
        hm.fit(X_tr,y_tr); h_pred=np.maximum(0,hm.predict(X_te))

        fm=XGBRegressor(subsample=0.8,colsample_bytree=0.8,
                        random_state=42,verbosity=0,**XGBOOST_PARAMS[fuel])
        fm.fit(X,y)
        fi=pd.Series(fm.feature_importances_,index=features).sort_values(ascending=False)

        results[fuel]={
            'cv_mapes':cv_mapes,'cv_maes':cv_maes,
            'cv_mape_mean':np.mean(cv_mapes),'cv_mae_mean':np.mean(cv_maes),
            'fold_preds':fp,'fold_actuals':fa,'fold_dates':fd,
            'h_pred':h_pred,'h_actual':y_te.values,
            'h_dates':dates.iloc[-20:].values,
            'h_mape':calc_mape(y_te.values,h_pred),
            'h_mae':mean_absolute_error(y_te,h_pred),
            'h_rmse':np.sqrt(mean_squared_error(y_te,h_pred)),
            'feat_imp':fi,'n_rows':len(X),
        }
        print(f"    CV:{np.mean(cv_mapes):.1f}%  Holdout:{results[fuel]['h_mape']:.1f}%  MAE:{results[fuel]['h_mae']:.1f}L")
    return results

# ─── FIGURE 1 ─────────────────────────────────────────────────────────────────

def fig1_overview(df):
    print("\n[1/6] Figure 1 — Dataset Overview")
    fig,axes=plt.subplots(2,2,figsize=(20,13))
    fig.patch.set_facecolor(BG)
    fig.suptitle('EMERALD LANKA — FUEL SALES PREDICTION MODEL\nDataset Overview & Sales Trends',
                 fontsize=16,fontweight='bold',color=GREEN,y=0.97)

    ax=axes[0,0]; sax(ax,'Daily Sales — All Fuels (14-Day Rolling Avg)','Date','Sales (L)')
    for fuel in FUELS:
        mask=df[fuel]>0
        roll=df[fuel].where(mask).rolling(14,center=True,min_periods=1).mean()
        ax.plot(df['date'],roll,color=FUEL_COLORS[fuel],lw=2,label=FUEL_LABELS[fuel],alpha=0.9)
    ax.legend(fontsize=8)

    ax=axes[0,1]; sax(ax,'Monthly Average Daily Sales','Month','Avg Sales (L)')
    x=np.arange(12); w=0.2
    for i,fuel in enumerate(FUELS):
        m_avg=[df[df['month']==m][fuel].replace(0,np.nan).mean() for m in range(1,13)]
        ax.bar(x+i*w,m_avg,w,color=FUEL_COLORS[fuel],alpha=0.85,
               label=FUEL_LABELS[fuel],edgecolor='#0B0D0C')
    ax.set_xticks(x+1.5*w); ax.set_xticklabels(MONTH_NAMES,fontsize=8); ax.legend(fontsize=8)

    ax=axes[1,0]; sax(ax,'Average Sales by Day of Week','Day','Avg Sales (L)')
    dow=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']; x=np.arange(7); w=0.35
    for i,fuel in enumerate(['petrol','diesel']):
        d_avg=[df[df['day_of_week']==d][fuel].replace(0,np.nan).mean() for d in range(7)]
        ax.bar(x+i*w,d_avg,w,color=FUEL_COLORS[fuel],alpha=0.85,
               label=FUEL_LABELS[fuel],edgecolor='#0B0D0C')
    ax.axvspan(4.5,6.5,alpha=0.08,color=WARNING,label='Weekend')
    ax.set_xticks(x+0.175); ax.set_xticklabels(dow,fontsize=9); ax.legend(fontsize=8)

    ax=axes[1,1]; sax(ax,'Daily Sales Distribution','Sales (L)','Frequency')
    for fuel in ['petrol','diesel']:
        vals=df[df[fuel]>0][fuel]
        ax.hist(vals,bins=35,alpha=0.6,color=FUEL_COLORS[fuel],
                label=FUEL_LABELS[fuel],edgecolor='#0B0D0C')
        ax.axvline(vals.mean(),color=FUEL_COLORS[fuel],lw=2,linestyle='--',
                   label=f'Avg: {vals.mean():.0f}L')
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0,0,1,0.95]); save(fig,'sf1_overview.png')

# ─── FIGURE 2 ─────────────────────────────────────────────────────────────────

def fig2_accuracy(res):
    print("[2/6] Figure 2 — Model Accuracy")
    fig,axes=plt.subplots(2,4,figsize=(22,11))
    fig.patch.set_facecolor(BG)
    fig.suptitle('MODEL ACCURACY — Actual vs Predicted (All 4 Fuels)\nTop: CV Scatter  |  Bottom: Holdout Time Series',
                 fontsize=13,fontweight='bold',color=GREEN,y=0.97)

    for col,fuel in enumerate(FUELS):
        r=res[fuel]; c=FUEL_COLORS[fuel]; lbl=FUEL_LABELS[fuel]

        ax=axes[0,col]; sax(ax,f'{lbl}\nActual vs Predicted (CV)','Actual (L)','Predicted (L)')
        ax.scatter(r['fold_actuals'],r['fold_preds'],alpha=0.3,color=c,s=12)
        mn=min(min(r['fold_actuals']),min(r['fold_preds']))
        mx=max(max(r['fold_actuals']),max(r['fold_preds']))
        ax.plot([mn,mx],[mn,mx],'--',color=WHITE,lw=1.5,label='Perfect')
        r2=np.corrcoef(r['fold_actuals'],r['fold_preds'])[0,1]**2
        ax.text(0.05,0.93,f'R²={r2:.3f}',transform=ax.transAxes,color=c,fontsize=9,fontweight='bold')
        ax.text(0.05,0.85,f'CV MAPE={r["cv_mape_mean"]:.1f}%',transform=ax.transAxes,color=WARNING,fontsize=8)
        ax.legend(fontsize=7)

        ax=axes[1,col]; sax(ax,f'{lbl}\nHoldout (Last 20 Days)','Date','Sales (L)')
        h_d=pd.to_datetime(r['h_dates'])
        ax.plot(h_d,r['h_actual'],'o-',color=c,lw=2,ms=4,label='Actual')
        ax.plot(h_d,r['h_pred'],'s--',color=WHITE,lw=1.5,ms=4,alpha=0.8,label='Predicted')
        ax.fill_between(h_d,r['h_actual'],r['h_pred'],alpha=0.15,color=WARNING)
        ax.text(0.02,0.92,f'MAPE={r["h_mape"]:.1f}%',transform=ax.transAxes,color=c,fontsize=9,fontweight='bold')
        ax.text(0.02,0.83,f'MAE={r["h_mae"]:.1f}L',transform=ax.transAxes,color=WARNING,fontsize=8)
        plt.setp(ax.xaxis.get_majorticklabels(),rotation=30,fontsize=7); ax.legend(fontsize=7)

    plt.tight_layout(rect=[0,0,1,0.95]); save(fig,'sf2_accuracy.png')

# ─── FIGURE 3 ─────────────────────────────────────────────────────────────────

def fig3_cv(res):
    print("[3/6] Figure 3 — CV Validation")
    fig,axes=plt.subplots(2,3,figsize=(20,12))
    fig.patch.set_facecolor(BG)
    fig.suptitle('CROSS-VALIDATION RESULTS — 5-Fold TimeSeriesSplit',
                 fontsize=14,fontweight='bold',color=GREEN,y=0.97)

    ax=axes[0,0]; sax(ax,'CV MAPE by Fold — Petrol & Diesel','Fold','MAPE (%)')
    x=np.arange(5); w=0.35
    for i,fuel in enumerate(['petrol','diesel']):
        ax.bar(x+i*w,res[fuel]['cv_mapes'],w,color=FUEL_COLORS[fuel],
               alpha=0.85,label=FUEL_LABELS[fuel],edgecolor='#0B0D0C')
        ax.axhline(np.mean(res[fuel]['cv_mapes']),color=FUEL_COLORS[fuel],lw=1.5,linestyle='--',alpha=0.7)
    ax.set_xticks(x+0.175); ax.set_xticklabels([f'Fold {i+1}' for i in range(5)]); ax.legend(fontsize=9)

    ax=axes[0,1]; sax(ax,'CV vs Holdout MAPE — All Fuels','Fuel','MAPE (%)')
    lbls=[FUEL_LABELS[f] for f in FUELS]
    x=np.arange(4); w=0.35
    b1=ax.bar(x-w/2,[res[f]['cv_mape_mean'] for f in FUELS],w,label='CV MAPE',color=BLUE,alpha=0.85)
    b2=ax.bar(x+w/2,[res[f]['h_mape']       for f in FUELS],w,label='Holdout MAPE',color=DKGREEN,alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(lbls,fontsize=8,rotation=10)
    for bar in list(b1)+list(b2):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,
                f'{bar.get_height():.1f}%',ha='center',color=WHITE,fontsize=7)
    ax.legend(fontsize=9)

    for col,(fuel,c) in enumerate(zip(['petrol','diesel'],[BLUE,WARNING])):
        ax=axes[0,2] if fuel=='petrol' else axes[1,0]
        sax(ax,f'{FUEL_LABELS[fuel]} Residuals (CV)','Actual (L)','Residual (L)')
        r=res[fuel]; resids=np.array(r['fold_preds'])-np.array(r['fold_actuals'])
        ax.scatter(r['fold_actuals'],resids,alpha=0.3,color=c,s=12)
        ax.axhline(0,color=GREEN,lw=2,linestyle='--')
        ax.axhline(resids.mean(),color=WARNING,lw=1,linestyle=':',label=f'Mean:{resids.mean():.1f}L')
        ax.fill_between([min(r['fold_actuals']),max(r['fold_actuals'])],
                        resids.std(),-resids.std(),alpha=0.08,color=c,
                        label=f'±1σ:{resids.std():.1f}L')
        ax.legend(fontsize=8)

    ax=axes[1,1]; sax(ax,'Error Distribution — Petrol','Error (L)','Frequency')
    r=res['petrol']; resids=np.array(r['fold_preds'])-np.array(r['fold_actuals'])
    ax.hist(resids,bins=35,color=BLUE,alpha=0.7,edgecolor='#0B0D0C')
    ax.axvline(0,color=GREEN,lw=2,label='Zero')
    ax.axvline(resids.mean(),color=WARNING,lw=1.5,linestyle='--',label=f'Mean:{resids.mean():.1f}L')
    ax.axvline(resids.mean()+resids.std(),color=ERROR,lw=1,linestyle=':')
    ax.axvline(resids.mean()-resids.std(),color=ERROR,lw=1,linestyle=':',label=f'±1σ:{resids.std():.1f}L')
    ax.legend(fontsize=8)

    ax=axes[1,2]; sax(ax,'Holdout MAE — All Fuels','Fuel','MAE (L)')
    bars=ax.bar(lbls,[res[f]['h_mae'] for f in FUELS],
                color=[FUEL_COLORS[f] for f in FUELS],alpha=0.85,edgecolor='#0B0D0C',width=0.5)
    for bar,val in zip(bars,[res[f]['h_mae'] for f in FUELS]):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f'{val:.1f}L',ha='center',color=WHITE,fontsize=9)
    ax.set_xticklabels(lbls,rotation=10,fontsize=9)

    plt.tight_layout(rect=[0,0,1,0.95]); save(fig,'sf3_cv_validation.png')

# ─── FIGURE 4 ─────────────────────────────────────────────────────────────────

def fig4_features(res):
    print("[4/6] Figure 4 — Feature Importance")
    fig,axes=plt.subplots(2,2,figsize=(18,12))
    fig.patch.set_facecolor(BG)
    fig.suptitle('FEATURE IMPORTANCE — Top Predictors per Fuel Model',
                 fontsize=14,fontweight='bold',color=GREEN,y=0.97)

    for idx,fuel in enumerate(FUELS):
        ax=axes[idx//2,idx%2]; r=res[fuel]; c=FUEL_COLORS[fuel]
        top=r['feat_imp'].head(12)
        labels=[f.replace('_',' ').title() for f in top.index]
        fc=[c if i==0 else DKGREEN if i<3 else DIM for i in range(len(top))]
        bars=ax.barh(labels[::-1],top.values[::-1],color=fc[::-1],alpha=0.85,edgecolor='#0B0D0C',height=0.65)
        for bar,val in zip(bars,top.values[::-1]):
            ax.text(bar.get_width()+0.002,bar.get_y()+bar.get_height()/2,f'{val:.4f}',va='center',color=WHITE,fontsize=8)
        sax(ax,f'{FUEL_LABELS[fuel]} — Top 12 Features','Importance','')
        ax.set_xlim(0,top.values.max()*1.25)

    plt.tight_layout(rect=[0,0,1,0.95]); save(fig,'sf4_features.png')

# ─── FIGURE 5 ─────────────────────────────────────────────────────────────────

def fig5_seasonal(df):
    print("[5/6] Figure 5 — Seasonal Patterns")
    fig,axes=plt.subplots(2,2,figsize=(18,12))
    fig.patch.set_facecolor(BG)
    fig.suptitle('SEASONAL & TEMPORAL PATTERNS IN FUEL SALES',
                 fontsize=14,fontweight='bold',color=GREEN,y=0.97)

    ax=axes[0,0]; sax(ax,'Monthly Sales Pattern — Petrol & Diesel','Month','Avg Sales (L)')
    for fuel in ['petrol','diesel']:
        m_avg=[df[df['month']==m][fuel].replace(0,np.nan).mean() for m in range(1,13)]
        m_std=[df[df['month']==m][fuel].replace(0,np.nan).std()  for m in range(1,13)]
        ax.plot(MONTH_NAMES,m_avg,'o-',color=FUEL_COLORS[fuel],lw=2.5,ms=7,label=FUEL_LABELS[fuel])
        ax.fill_between(MONTH_NAMES,[v-s for v,s in zip(m_avg,m_std)],
                        [v+s for v,s in zip(m_avg,m_std)],alpha=0.12,color=FUEL_COLORS[fuel])
    ax.axvspan(4,8,alpha=0.07,color=BLUE,label='Monsoon'); ax.legend(fontsize=9)
    ax.set_xticklabels(MONTH_NAMES,fontsize=8)

    ax=axes[0,1]; sax(ax,'Day-of-Week Sales Pattern','Day','Avg Sales (L)')
    dow=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    for fuel in ['petrol','diesel']:
        d_avg=[df[df['day_of_week']==d][fuel].replace(0,np.nan).mean() for d in range(7)]
        ax.plot(dow,d_avg,'o-',color=FUEL_COLORS[fuel],lw=2.5,ms=7,label=FUEL_LABELS[fuel])
    ax.axvspan(4.5,6.5,alpha=0.08,color=WARNING,label='Weekend'); ax.legend(fontsize=9)

    ax=axes[1,0]; sax(ax,'30-Day Rolling Sales Trend','Date','Sales (L)')
    for fuel in ['petrol','diesel']:
        mask=df[fuel]>0
        roll=df[fuel].where(mask).rolling(30,center=True,min_periods=1).mean()
        ax.plot(df['date'],roll,color=FUEL_COLORS[fuel],lw=2.5,label=FUEL_LABELS[fuel],alpha=0.9)
    for m_val in [5,6,7,8,9]:
        mask=df['month']==m_val
        if mask.any(): ax.axvspan(df['date'][mask].min(),df['date'][mask].max(),alpha=0.04,color=BLUE)
    ax.legend(fontsize=9)

    ax=axes[1,1]; sax(ax,'Petrol Sales by Quarter','Quarter','Sales (L)')
    groups=[df[(df['quarter']==q)&(df['petrol']>0)]['petrol'].values for q in range(1,5)]
    qc=[WARNING,BLUE,DKGREEN,PURPLE]
    bp=ax.boxplot(groups,patch_artist=True,
                  medianprops=dict(color='#0B0D0C',lw=2),
                  whiskerprops=dict(color=DIM),capprops=dict(color=DIM),
                  flierprops=dict(marker='o',color=ERROR,alpha=0.4,markersize=3))
    for patch,c in zip(bp['boxes'],qc): patch.set_facecolor(c); patch.set_alpha(0.75)
    ax.set_xticklabels(['Q1','Q2','Q3','Q4'],fontsize=10)
    for i,grp in enumerate(groups):
        if len(grp)>0: ax.text(i+1,np.median(grp)+30,f'Med:{np.median(grp):.0f}L',ha='center',color=WHITE,fontsize=8)

    plt.tight_layout(rect=[0,0,1,0.95]); save(fig,'sf5_seasonal.png')

# ─── FIGURE 6 ─────────────────────────────────────────────────────────────────

def fig6_summary(res):
    print("[6/6] Figure 6 — Performance Summary")
    fig,ax=plt.subplots(figsize=(18,8))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
    fig.suptitle('MODEL PERFORMANCE SUMMARY — EMERALD LANKA FUEL SALES PREDICTOR',
                 fontsize=16,fontweight='bold',color=GREEN,y=0.97)

    metrics=[
        ('92 Petrol',   'XGBoost',res['petrol']['n_rows'],res['petrol']['cv_mape_mean'],      res['petrol']['h_mape'],      res['petrol']['h_mae'],      res['petrol']['h_rmse'],      'High',  GREEN),
        ('95 Petrol',   'XGBoost',res['super_petrol']['n_rows'],res['super_petrol']['cv_mape_mean'],res['super_petrol']['h_mape'],res['super_petrol']['h_mae'],res['super_petrol']['h_rmse'],'Low',   ERROR),
        ('Auto Diesel', 'XGBoost',res['diesel']['n_rows'],res['diesel']['cv_mape_mean'],       res['diesel']['h_mape'],      res['diesel']['h_mae'],      res['diesel']['h_rmse'],      'Medium',WARNING),
        ('Super Diesel','XGBoost',res['super_diesel']['n_rows'],res['super_diesel']['cv_mape_mean'],res['super_diesel']['h_mape'],res['super_diesel']['h_mae'],res['super_diesel']['h_rmse'],'Low',   ERROR),
    ]
    headers=['Fuel Type','Algorithm','Train Rows','CV MAPE','Holdout MAPE','MAE (L)','RMSE (L)','Reliability']
    col_x=[0.02,0.17,0.30,0.42,0.55,0.67,0.79,0.90]
    row_h=0.11; sy=0.82

    for header,x in zip(headers,col_x):
        ax.text(x,sy+0.04,header,transform=ax.transAxes,color=GREEN,fontsize=9,fontweight='bold',va='center')
    ax.plot([0.01,0.99],[sy,sy],transform=ax.transAxes,color=GREEN,lw=1)

    for i,(fuel,algo,rows,cv,ho,mae,rmse,rel,c) in enumerate(metrics):
        y=sy-(i+1)*row_h; bg_c='#1A2020' if i%2==0 else SURFACE
        rect=FancyBboxPatch((0.01,y-0.02),0.98,row_h-0.01,boxstyle='round,pad=0.005',
                            facecolor=bg_c,edgecolor='#2A3530',transform=ax.transAxes,linewidth=0.5)
        ax.add_patch(rect)
        vals=[fuel,algo,str(rows),f'{cv:.1f}%',f'{ho:.1f}%',f'{mae:.1f}',f'{rmse:.1f}',rel]
        colors=[WHITE,DIM,DIM,WARNING,c,DIM,DIM,c]
        for val,x2,col in zip(vals,col_x,colors):
            ax.text(x2,y+row_h/2-0.02,val,transform=ax.transAxes,color=col,fontsize=9,
                    va='center',fontweight='bold' if val in [fuel,rel] else 'normal')

    summary=[
        ('Training Data', '385 days (Apr 2025 – Apr 2026)'),
        ('Algorithm',     'XGBoost Gradient Boosting'),
        ('Validation',    '5-fold TimeSeriesSplit (no data leakage)'),
        ('Best Models',   'Petrol & Diesel — reliable for stock ordering'),
        ('Limitations',   '95 Petrol & Super Diesel: high OOS rate'),
        ('Key Features',  'Lag-7, rolling avg, seasonal & day-of-week'),
    ]
    for j,(lbl,val) in enumerate(summary):
        x_p=0.02+(j%3)*0.33; y_p=0.08 if j>=3 else 0.17
        ax.text(x_p,y_p,f'{lbl}:',transform=ax.transAxes,color=DIM,fontsize=9)
        ax.text(x_p+0.12,y_p,val,transform=ax.transAxes,color=WHITE,fontsize=9,fontweight='bold')

    save(fig,'sf6_summary.png')

# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n"+"="*60)
    print("  SALES MODEL EVALUATION — EMERALD LANKA")
    print("="*60)

    print("\n[0] Loading sales data...")
    df = load_data("data/sales_data_clean.csv")

    print("\n[Eval] Running 5-fold CV + holdout for all 4 fuels...")
    results = evaluate_all(df)

    print(f"\n  Generating figures → {OUTPUT_DIR}/")
    fig1_overview(df)
    fig2_accuracy(results)
    fig3_cv(results)
    fig4_features(results)
    fig5_seasonal(df)
    fig6_summary(results)

    print(f"\n{'='*60}")
    print(f"  DONE — 6 figures saved to {OUTPUT_DIR}/")
    print(f"{'='*60}")
    print(f"\n  {'Fuel':<20} {'CV MAPE':>10} {'Holdout':>10} {'MAE':>10}")
    print(f"  {'-'*52}")
    for fuel in FUELS:
        r=results[fuel]
        print(f"  {FUEL_LABELS[fuel]:<20} {r['cv_mape_mean']:>9.1f}% {r['h_mape']:>9.1f}% {r['h_mae']:>9.1f}L")
