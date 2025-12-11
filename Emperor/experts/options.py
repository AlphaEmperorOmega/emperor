from Emperor.base.enums import BaseOptions
from Emperor.experts.utils.layers import MixtureOfExperts
from Emperor.experts.utils.stack import MixtureOfExpertsStack


class MixtureOfExpertsOptions(BaseOptions):
    BASE = MixtureOfExperts


class MixtureOfExpertsStackOptions(BaseOptions):
    BASE = MixtureOfExpertsStack
