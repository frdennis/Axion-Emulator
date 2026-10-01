import torch
import jax.numpy as jnp

class GlobalStandardScaler:
    def __init__(self, mean_=None, scale_=None, flax: bool = False):
        self.flax = flax
        if mean_ is not None:
            self.mean_ = mean_
            if flax:
                self.mean_ = jnp.asarray(self.mean_)
        if scale_ is not None:
            self.scale_ = scale_
            if flax:
                self.scale_ = jnp.asarray(self.scale_)

    def _reset(self):
        if hasattr(self, "scale_"):
            del self.scale_
            del self.mean_

    def fit(self, X):
        self._reset()
        self.mean_ = X.mean()
        self.scale_ = X.std()

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        return self.scale_ * X + self.mean_

    def to_dict(self,):
        return {"type": self.__class__.__name__, "attrs": vars(self)}


class StandardScaler:
    def __init__(self, mean_=None, scale_=None, flax: bool = False):
        self.flax = flax
        if mean_ is not None:
            self.mean_ = mean_
            if flax:
                self.mean_ = jnp.asarray(self.mean_)
        if scale_ is not None:
            self.scale_ = scale_
            if flax:
                self.scale_ = jnp.asarray(self.scale_)

    def _reset(self):
        if hasattr(self, "scale_"):
            del self.scale_
            del self.mean_

    def fit(self, X):
        self._reset()
        self.mean_ = X.mean(0, keepdim=True)
        self.scale_ = X.std(0, unbiased=False, keepdim=True)
        self.scale_[self.scale_ == 0.0] = 1.0

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        return self.scale_ * X + self.mean_

    def to_dict(self,):
        return {"type": self.__class__.__name__, "attrs": vars(self)}


class MinMaxScaler:
    def __init__(self, min_=None, max_=None, flax: bool = False):
        self.flax = flax
        if min_ is not None:
            self.min_ = min_
            if flax:
                self.min_ = jnp.asarray(self.min_)
        if max_ is not None:
            self.max_ = max_
            if flax:
                self.max_= jnp.asarray(self.max_)

    def _reset(self):
        if hasattr(self, "min_"):
            del self.min_
            del self.max_

    def fit(self, X):
        self._reset()
        self.min_, _ = X.min(0, keepdim=True)
        self.max_, _ = X.max(0, keepdim=True)
        self.max_[(self.min_ == 0.) & (self.max_ == 0.)] = 1.

    def transform(self, X):
        return (X - self.min_) / (self.max_ - self.min_)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        return (self.max_ - self.min_) * X + self.min_

    def to_dict(self,):
        return {"type": self.__class__.__name__, "attrs": vars(self)}



class LogScaler:
    def __init__(self, flax: bool = False):
        self.flax = flax

    def fit(self, X):
        pass

    def transform(self, X):
        if self.flax:
            return jnp.log(X)
        return torch.log(X)

    def fit_transform(self, X):
        return self.transform(X)

    def inverse_transform(self, X):
        if self.flax:
            return jnp.exp(X)
        return torch.exp(X)

    def to_dict(self,):
        return {"type": self.__class__.__name__, "attrs": vars(self)}

class Log10Scaler:
    def __init__(self, flax: bool = False, **kwargs):
        self.flax = flax
        self.min_value = -1.01

    def fit(self, X):
        pass

    def transform(self, X):
        if self.flax:
            return jnp.log10(X - self.min_value)
        return torch.log10(X-self.min_value)

    def fit_transform(self, X):
        return self.transform(X)

    def inverse_transform(self, X):
        return 10**X  + self.min_value

    def to_dict(self,):
        return {"type": self.__class__.__name__, "attrs": vars(self)}


class IdentityScaler:
    def __init__(self, **kwargs):
        pass

    def fit(self, X):
        pass

    def transform(self, X):
        return X

    def fit_transform(self, X):
        return X

    def inverse_transform(self, X):
        return X

    def to_dict(self,):
        return {"type": self.__class__.__name__, "attrs": vars(self)}


class StandardLogScaler(StandardScaler):
    def __init__(self, mean_=None, scale_=None, flax=False):
        super().__init__(mean_=mean_, scale_=scale_, flax=flax)

    def transform(self, X):
        if self.flax:
            X = jnp.log10(X)
        else:
            X = torch.log10(X)
        return super().transform(X)

    def fit(self, X):
        if self.flax:
            X = jnp.log10(X)
        else:
            X = torch.log10(X)
        return super().fit(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X_destandarised = super().inverse_transform(X)
        if self.flax:
            return 10**(X_destandarised)
        return 10**(X_destandarised)


class SpecialStandardLogScaler(StandardScaler):
    def __init__(self, mean_=None, scale_=None, flax: bool = False,**kwargs):
        super().__init__(mean_=mean_, scale_=scale_, flax=flax)
        self.min_value = -1.01 # TODO: Check varying this

    def transform(self, X):
        if self.flax:
            X = jnp.log10(X - self.min_value)
        else:
            X = torch.log10(X - self.min_value)
        return super().transform(X)

    def fit(self, X):
        if self.flax:
            X = jnp.log10(X - self.min_value)
        else:
            X = torch.log10(X - self.min_value)
        return super().fit(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X_destandarised = super().inverse_transform(X)
        if self.flax:
            return 10**(X_destandarised) + self.min_value
        return 10**(X_destandarised) + self.min_value

class GlobalSpecialStandardLogScaler(GlobalStandardScaler):
    def __init__(self, mean_=None, scale_=None, flax: bool = False,**kwargs):
        super().__init__(mean_=mean_, scale_=scale_, flax=flax)
        self.min_value = -1.01 # TODO: Check varying this

    def transform(self, X):
        if self.flax:
            X = jnp.log10(X - self.min_value)
        else:
            X = torch.log10(X - self.min_value)
        return super().transform(X)

    def fit(self, X):
        if self.flax:
            X = jnp.log10(X - self.min_value)
        else:
            X = torch.log10(X - self.min_value)
        return super().fit(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X_destandarised = super().inverse_transform(X)
        if self.flax:
            return 10**(X_destandarised) + self.min_value
        return 10**(X_destandarised) + self.min_value
