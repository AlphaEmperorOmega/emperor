from emperor.base.enums import BaseOptions
from emperor.experts.utils.layers import MixtureOfExperts
from emperor.experts.utils.model import MixtureOfExpertsModel


class MixtureOfExpertsOptions(BaseOptions):
    BASE = MixtureOfExperts


class MixtureOfExpertsStackOptions(BaseOptions):
    BASE = MixtureOfExpertsModel
