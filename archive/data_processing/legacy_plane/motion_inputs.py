"""Label-free interval-motion inputs; calibration fits train-clean only.

The six CAFAD input channels are explicitly adapted, not unchanged paper data:
east/north GPS-vs-reported velocity residuals, altitude vertical rate,
pressure-vs-altitude rate residual, speed acceleration and wrapped turn rate.
"""
import numpy as np


class MotionAdapter:
    def __init__(self,config):
        self.metadata=dict(config)
        self.cache_mean=np.asarray(config['cache_mean'],dtype=np.float64)
        self.cache_scale=np.asarray(config['cache_scale'],dtype=np.float64)
        self.speed_factor=float(config['speed_factor'])
        self.pressure_factor=float(config['pressure_factor'])
        self.center=np.asarray(config['center'],dtype=np.float64)
        self.robust_scale=np.asarray(config['robust_scale'],dtype=np.float64)
        self.mean=np.asarray(config['transformed_mean'],dtype=np.float64)
        self.scale=np.asarray(config['transformed_scale'],dtype=np.float64)
        if any(a.shape!=(6,) or not np.isfinite(a).all() for a in [self.cache_mean,self.cache_scale,self.center,self.robust_scale,self.mean,self.scale]):
            raise ValueError('Motion adapter needs six finite channel scales')
        if np.any(self.scale<=0) or np.any(self.robust_scale<=0) or np.any(self.cache_scale<=0):raise ValueError('Motion scales must be positive')
        if not np.isfinite([self.speed_factor,self.pressure_factor]).all() or self.speed_factor<=0:raise ValueError('Invalid train calibration')

    def _intervals(self,features,elapsed,mask):
        shape=features.shape
        values=np.asarray(features,dtype=np.float64).reshape(-1,shape[-2],6)*self.cache_scale+self.cache_mean
        valid=np.asarray(mask,dtype=bool).reshape(-1,shape[-2]);times=np.asarray(elapsed,dtype=np.float64).reshape(valid.shape)
        positions=np.broadcast_to(np.arange(shape[-2]),valid.shape)
        preceding=np.maximum.accumulate(np.where(valid,positions,-1),axis=1)
        previous=np.pad(preceding[:,:-1],((0,0),(1,0)),constant_values=-1)
        rows=np.arange(len(values))[:,None];safe=previous.clip(min=0)
        before=values[rows,safe];dt=times-times[rows,safe];usable=valid&(previous>=0)
        if np.any(dt[usable]<=0):raise ValueError('Motion derivatives require positive elapsed intervals')
        dt=np.where(usable,dt,1.)
        dlon=(values[...,0]-before[...,0]+180.)%360.-180.
        midlat=np.deg2rad((values[...,1]+before[...,1])*.5)
        vx=6371000.*np.cos(midlat)*np.deg2rad(dlon)/dt
        vy=6371000.*np.deg2rad(values[...,1]-before[...,1])/dt
        reported_x=.5*(values[...,4]*np.sin(values[...,5])+before[...,4]*np.sin(before[...,5]))
        reported_y=.5*(values[...,4]*np.cos(values[...,5])+before[...,4]*np.cos(before[...,5]))
        altitude=values[...,2]-before[...,2];pressure=values[...,3]-before[...,3]
        heading=(values[...,5]-before[...,5]+np.pi)%(2*np.pi)-np.pi
        derived=np.stack([vx-self.speed_factor*reported_x,vy-self.speed_factor*reported_y,
            altitude/dt,(pressure-self.pressure_factor*altitude)/dt,
            (values[...,4]-before[...,4])/dt,heading/dt],axis=-1)
        derived=np.where(usable[...,None],derived,0.)
        return derived.reshape(shape),usable.reshape(mask.shape),{
            'gps_speed':np.hypot(vx,vy),'reported_speed':np.abs(.5*(values[...,4]+before[...,4])),
            'altitude_difference':altitude,'pressure_difference':pressure}

    @classmethod
    def fit(cls,data,variants):
        scaler=data.manifest['scaler']
        config={'kind':'motion_consistency','cache_mean':scaler['mean'],'cache_scale':scaler['scale'],
            'speed_factor':1.,'pressure_factor':1.,'center':[0.]*6,'robust_scale':[1.]*6,
            'transformed_mean':[0.]*6,'transformed_scale':[1.]*6}
        initial=cls(config)
        selected=np.flatnonzero((data.split[data.flight_index]=='train')&(variants[data.flight_index]=='clean'))
        if not len(selected):raise ValueError('Train-clean calibration pool is empty')
        features,elapsed,mask=data.features[selected],data.elapsed[selected],data.mask[selected]
        _,usable,intervals=initial._intervals(features,elapsed,mask)
        eligible=usable&(intervals['reported_speed']>1.)&(intervals['gps_speed']>0.)
        ratios=intervals['gps_speed'][eligible]/intervals['reported_speed'][eligible]
        if not len(ratios) or not np.isfinite(ratios).all():raise ValueError('No finite speed calibration intervals')
        config['speed_factor']=float(np.median(ratios))
        height=usable&(np.abs(intervals['altitude_difference'])>1.)
        if not height.any():raise ValueError('No finite altitude calibration intervals')
        config['pressure_factor']=float(np.median(intervals['pressure_difference'][height]/intervals['altitude_difference'][height]))
        calibrated=cls(config);derived,usable,_=calibrated._intervals(features,elapsed,mask)
        fitting=derived[usable]
        center=np.median(fitting,axis=0);robust=1.4826*np.median(np.abs(fitting-center),axis=0)
        robust=np.maximum(robust,1e-6)
        transformed=np.arcsinh((fitting-center)/robust)
        mean=transformed.mean(0);scale=transformed.std(0);scale[scale<1e-8]=1.
        config.update(center=center.tolist(),robust_scale=robust.tolist(),transformed_mean=mean.tolist(),transformed_scale=scale.tolist(),
            fitting_intervals=len(fitting),speed_calibration_intervals=int(eligible.sum()),pressure_calibration_intervals=int(height.sum()),
            fit_split='train_clean_only',feature_names=['east_velocity_residual','north_velocity_residual','altitude_vertical_rate',
                'pressure_altitude_rate_residual','speed_acceleration','wrapped_heading_rate'],
            mapping='Robust train-clean center/MAD(1e-6 floor), asinh, then train-clean mean/std; first observation has no preceding interval and outputs zero',
            label_or_identity_feature=False,own_clean_test_counterpart_used=False,anchors_and_contexts_changed=False,
            calibration_scope='Empirical scale conversions from training observations, not externally certified source physical units')
        return cls(config)

    def transform(self,features,elapsed,mask):
        values,usable,_=self._intervals(features,elapsed,mask)
        transformed=(np.arcsinh((values-self.center)/self.robust_scale)-self.mean)/self.scale
        result=np.where(usable[...,None],transformed,0.).astype(np.float32)
        if not np.isfinite(result).all():raise FloatingPointError('Motion adapter produced nonfinite inputs')
        return result

    def transform_batch(self,batch):
        result=dict(batch)
        result['features']=self.transform(batch['features'],batch['elapsed'],batch['mask'])
        result['peers']=self.transform(batch['peers'],batch['peer_elapsed'],batch['peer_mask'])
        return result

    def config(self):return dict(self.metadata)

    @classmethod
    def from_config(cls,config):return cls(config)
