import os
from os.path import exists
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource


class yml_config_setting(PydanticBaseSettingsSource):

    __config: Dict[str, Any] = {}

    env_prefix: str = "CCDW_CFG"

    def load_yml_file(self) -> Dict[str, Any]:
        used_env = "<None Found>"

        if (self.__config == {}):
            config = {
                f"{self.env_prefix}_FULL_PATH": "",
                f"{self.env_prefix}_CFG_PATH": ".",
                f"{self.env_prefix}_CFG_FN": "config.yml",
                **os.environ,
                **dotenv_values(f"{self.env_prefix}_FULL_PATH"),
                **dotenv_values(f"{self.env_prefix}_PATH"),
                **dotenv_values(f"{self.env_prefix}_FN"),
            }

            if f"{self.env_prefix}_FULL_PATH" in config and config[f"{self.env_prefix}_FULL_PATH"] != "":
                config_file = config[f"{self.env_prefix}_FULL_PATH"]
                used_env = f"{self.env_prefix}_FULL_PATH"
            elif f"{self.env_prefix}_FN" in config and config[f"{self.env_prefix}_FN"] != "":
                if f"{self.env_prefix}_PATH" in config and config[f"{self.env_prefix}_PATH"] != "":
                    used_env = f"{self.env_prefix}_PATH and {self.env_prefix}_FN"
                    config_file = os.path.join(
                        config[f"{self.env_prefix}_PATH"], config[f"{self.env_prefix}_FN"]
                    )
                else:
                    used_env = f"{self.env_prefix}_FN"
                    config_file = os.path.join(".", config[f"{self.env_prefix}_FN"])
            else:
                config_file = os.path.join(".", "config.yml")

            if exists(config_file):
                with open(config_file, "r") as f:
                    config_dict = yaml.safe_load(f)
                config_dict["config"]["location"] = config_file
            else:
                config_dict = {}
                raise Exception(f"The configuration file [{config_file}] does not exist. Used environment variable [{used_env}].")

            __config = config_dict

        return __config

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> Tuple[Any, str, bool]:
        config_dict = self.load_yml_file()
        field_value = config_dict[field_name]
        return field_value, field_name, False

    def prepare_field_value(
        self, field_name: str, field: FieldInfo, value: Any, value_is_complex: bool
    ) -> Any:
        return value

    def __call__(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}

        for field_name, field in self.settings_cls.model_fields.items():
            field_value, field_key, value_is_complex = self.get_field_value(
                field, field_name
            )
            field_value = self.prepare_field_value(
                field_name, field, field_value, value_is_complex
            )
            if field_value is not None:
                d[field_key] = field_value

        return d

class SchoolModel(BaseModel):
    name: Optional[str] = ""
    abbrev: Optional[str] = ""
    ipeds: Optional[Union[str,int]] = ""
    fice: Optional[str] = ""
    usgov: Optional[int] = None
    ncccs: Optional[int] = None
    instid: Optional[str] = ""
    inststate: Optional[str] = ""
    instcountry: Optional[str] = ""
    branch: Optional[str] = ""


class SQLModel(BaseModel):
    server: Optional[str] = ""
    db: Optional[str] = ""
    driver: Optional[str] = ""
    schema_input: Optional[str] = ""
    schema_history: Optional[str] = ""
    schema_local: Optional[str] = ""
    schema_ccdw: Optional[str] = ""
    schema_audit: Optional[str] = ""
    merge_scd1: Optional[str] = ""
    merge_scd2: Optional[str] = ""
    add_Columns: Optional[str] = ""
    drop_view: Optional[str] = ""
    delete_table_data: Optional[str] = ""
    view_create: Optional[str] = ""
    view2_create: Optional[str] = ""
    view2_cast: Optional[str] = ""
    view2_crossapply: Optional[str] = ""
    view2_whereand: Optional[str] = ""
    view3_create: Optional[str] = ""
    table_names: Optional[str] = ""
    table_column_names: Optional[str] = ""
    alter_table_keys: Optional[str] = ""
    alter_table_key_column: Optional[str] = ""
    alter_table_add: Optional[str] = ""
    alter_table_column: Optional[str] = ""
    audit_create_record: Optional[str] = ""
    audit_update_record: Optional[str] = ""


class Informer(BaseModel):
    export_path: Optional[str] = ""
    export_path_wStatus: Optional[str] = ""
    export_path_meta: Optional[str] = ""
    stage_path: Optional[str] = ""
    prefix: Optional[str] = ""
    latest_version: Optional[Union[str,int]] = ""


class CCDWModel(BaseModel):
    ccdw_path: Optional[str] = ""
    log_path: Optional[str] = ""
    archive_path: Optional[str] = ""
    archive_path_wStatus: Optional[str] = ""
    invalid_path_wStatus: Optional[str] = ""
    archive_type: Optional[str] = ""
    error_path: Optional[str] = ""
    error_output: Optional[bool] = False
    log_level: Optional[str] = "info"
    meta_custom: Optional[str] = ""
    new_fields_fn: Optional[str] = ""


class StatusFieldsModel(BaseModel):
    ACAD_PROGRAMS: Optional[Union[List[str],str]] = ""
    APPLICATIONS: Optional[Union[List[str],str]] = ""
    COURSES: Optional[Union[List[str],str]] = ""
    STUDENT_ACAD_CRED: Optional[Union[List[str],str]] = ""
    STUDENT_PROGRAMS: Optional[Union[List[str],str]] = ""
    STUDENT_TERMS: Optional[Union[List[str],str]] = ""
    XCC_ACAD_PROGRAM_REQMTS: Optional[Union[List[str],str]] = ""


class PyColleagueModel(BaseModel):
    source: Optional[str] = "ccdw"
    sourcepath: Optional[str] = "./input"


class DatamartModel(BaseModel):
    rootfolder: Optional[str] = ""


class RModel(BaseModel):
    scripts_path: Optional[str] = ""


#class ConfigModel(BaseSettings):
class CCDWConfigModel(BaseModel):
    location: Optional[str] = Field(validation_alias="CCDW_CFG_FULL_PATH")
    location_fn: Optional[str] = Field(validation_alias="CCDW_CFG_FN")
    location_path: Optional[str] = Field(validation_alias="CCDW_CFG_PATH")


class Settings(BaseSettings, case_sensitive = False):
    school: Optional[SchoolModel] = SchoolModel()
    sql: Optional[SQLModel] = SQLModel()
    informer: Optional[Informer] = Informer()
    ccdw: Optional[CCDWModel] = CCDWModel()
    status_fields: Optional[StatusFieldsModel] = StatusFieldsModel()
    pycolleague: Optional[PyColleagueModel] = PyColleagueModel()
    datamart: Optional[DatamartModel] = DatamartModel()
    R: Optional[RModel] = RModel()
    #config: Optional[CCDWConfigModel] = CCDWConfigModel()

    model_config = SettingsConfigDict(
        env_file = ".env",
        arbitrary_types_allowed = True,
        validate_default = False,
        extra = "allow"
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return init_settings, env_settings, dotenv_settings, yml_config_setting(settings_cls)

def get_config():
    return Settings().model_dump()

if __name__ == "__main__":
    testdict = Settings().model_dump()

    print(testdict)
    print("Done")
