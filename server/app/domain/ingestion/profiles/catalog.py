"""Built-in profiles registered by the default application composition."""

from domain.ingestion.profiles.contract import ContractDomainProfile
from domain.ingestion.profiles.decree import DecreeDomainProfile
from domain.ingestion.profiles.general import GeneralDomainProfile
from domain.ingestion.profiles.legal import LegalDomainProfile


SETTINGS_PROFILE_TYPES = (LegalDomainProfile, DecreeDomainProfile, ContractDomainProfile)
PROFILE_TYPES = (GeneralDomainProfile, *SETTINGS_PROFILE_TYPES)
