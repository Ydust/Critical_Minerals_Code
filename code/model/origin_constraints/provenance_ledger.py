"""Partial provenance, without forced dilution of observed lower amounts."""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Scope:
    entity: str
    year: int
    product: str
    unit: str
    provenance: str


@dataclass
class Ledger:
    scope: Scope
    total: float
    known_lower: dict[str, float]

    def __post_init__(self):
        if not isfinite(self.total) or self.total<=0:
            raise ValueError('A positive matched denominator is required.')
        if any(not isfinite(x) or x<0 for x in self.known_lower.values()):
            raise ValueError('Origin lower amounts must be finite and nonnegative.')
        if sum(self.known_lower.values())>self.total+1e-10:
            raise ValueError('Known amounts exceed the denominator.')
        if any(k in ('UNKNOWN','UNK','unresolved') for k in self.known_lower):
            raise ValueError('Unresolved provenance is not an origin country.')

    @property
    def unresolved(self):
        return max(0.,self.total-sum(self.known_lower.values()))

    def require_scope(self,requested:Scope):
        if requested!=self.scope:
            raise ValueError('Evidence scope does not match requested entity/year/product/unit/provenance.')

    def share_interval(self,origin):
        known=self.known_lower.get(origin,0.)
        return known/self.total,(known+self.unresolved)/self.total

    def hhi_outer_bounds(self):
        shares=[x/self.total for x in self.known_lower.values()]
        residual=self.unresolved/self.total
        lower=sum(x*x for x in shares)
        largest=max(shares,default=0.)
        return lower,lower+2*largest*residual+residual**2

    def complete_residual(self,prior):
        if not prior or any(not isfinite(x) or x<0 for x in prior.values()):
            raise ValueError('A nonnegative normalized scenario prior is required.')
        if any(k in ('UNKNOWN','UNK','unresolved') for k in prior):
            raise ValueError('Unresolved provenance is not a country in a completion scenario.')
        if abs(sum(prior.values())-1)>1e-10:
            raise ValueError('Scenario prior must sum to one.')
        completed={k:v/self.total for k,v in self.known_lower.items()}
        for origin,share in prior.items():
            completed[origin]=completed.get(origin,0.)+self.unresolved/self.total*share
        return completed


def product_bound(total_low,total_high,own_low,own_high,product_share_low,product_share_high):
    if not (0<total_low<=total_high and 0<=own_low<=own_high<=total_low and 0<product_share_low<=product_share_high<=1):
        raise ValueError('Invalid or incompatible rounded source bounds.')
    known_amount_low=max(0.,own_low-(1-product_share_low)*total_high)
    own_fraction_low=max(0.,1-(total_high-own_low)/(product_share_low*total_high))
    return dict(product_ni_low=product_share_low*total_low,product_ni_high=product_share_high*total_high,
        own_feed_ni_lower=known_amount_low,own_feed_share_lower=own_fraction_low,
        russian_feed_share_upper=1.,country_hhi_outer_lower=own_fraction_low**2,country_hhi_outer_upper=1.)


def export_lower(export_ni_lower,unassigned_supply_upper,additional_allowance):
    if export_ni_lower<=0 or min(unassigned_supply_upper,additional_allowance)<0:
        raise ValueError('Invalid matched mass-balance inputs.')
    return max(0.,1-(unassigned_supply_upper+additional_allowance)/export_ni_lower)
