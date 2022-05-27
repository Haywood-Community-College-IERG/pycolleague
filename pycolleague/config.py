import os
from os.path import exists
from typing import Any, List, Optional

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, BaseSettings, Field
from pydantic.env_settings import SettingsSourceCallable


def yml_config_setting(settings: BaseSettings) -> dict[str, Any]:
    config = {
        "HAYWOODCC_CFG_FULL_PATH": "",
        "HAYWOODCC_CFG_PATH": ".",
        "HAYWOODCC_CFG_FN": "config.yml",
        **os.environ,
        **dotenv_values("HAYWOODCC_CFG_FULL_PATH"),
        **dotenv_values("HAYWOODCC_CFG_PATH"),
        **dotenv_values("HAYWOODCC_CFG_FN"),
    }

    if config["HAYWOODCC_CFG_FULL_PATH"] != "":
        config_file = config["HAYWOODCC_CFG_FULL_PATH"]
    elif config["HAYWOODCC_CFG_FN"] != "":
        if config["HAYWOODCC_CFG_PATH"] != "":
            config_file = os.path.join(
                config["HAYWOODCC_CFG_PATH"], config["HAYWOODCC_CFG_FN"]
            )
        else:
            config_file = os.path.join(".", config["HAYWOODCC_CFG_FN"])
    else:
        config_file = os.path.join(".", "config.yml")

    if exists(config_file):
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)
        config_dict["config"]["location"] = config_file
    else:
        config_dict = {}
    return config_dict


class SchoolModel(BaseModel):
    name: Optional[str] = ""
    abbrev: Optional[str] = ""
    ipeds: Optional[str] = ""
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
    latest_version: Optional[str] = ""


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
    ACAD_PROGRAMS: Optional[List[str]] = ""
    APPLICATIONS: Optional[List[str]] = ""
    COURSES: Optional[List[str]] = ""
    STUDENT_ACAD_CRED: Optional[List[str]] = ""
    STUDENT_PROGRAMS: Optional[List[str]] = ""
    STUDENT_TERMS: Optional[List[str]] = ""
    XCC_ACAD_PROGRAM_REQMTS: Optional[List[str]] = ""


class PyColleagueModel(BaseModel):
    source: Optional[str] = "ccdw"
    sourcepath: Optional[str] = "./input"


class DatamartModel(BaseModel):
    rootfolder: Optional[str] = ""


class RModel(BaseModel):
    scripts_path: Optional[str] = ""


class ConfigModel(BaseSettings):
    location: Optional[str] = Field(env="HAYWOODCC_CFG_FULL_PATH")
    location_fn: Optional[str] = Field(env="HAYWOODCC_CFG_FN")
    location_path: Optional[str] = Field(env="HAYWOODCC_CFG_PATH")


class Settings(BaseSettings):
    school: Optional[SchoolModel] = SchoolModel()
    sql: Optional[SQLModel] = SQLModel()
    informer: Optional[Informer] = Informer()
    ccdw: Optional[CCDWModel] = CCDWModel()
    status_fields: Optional[StatusFieldsModel] = StatusFieldsModel()
    pycolleague: Optional[PyColleagueModel] = PyColleagueModel()
    datamart: Optional[DatamartModel] = DatamartModel()
    R: Optional[RModel] = RModel()
    config: Optional[ConfigModel] = ConfigModel()

    class Config:
        env_file: str = ".env"
        case_sensitive: bool = False
        arbitrary_types_allowed: bool = True
        validate_all: bool = False

        @classmethod
        def customise_sources(
            cls,
            init_settings: SettingsSourceCallable,
            env_settings: SettingsSourceCallable,
            file_secret_settings: SettingsSourceCallable,
        ) -> tuple[SettingsSourceCallable, ...]:
            return (
                init_settings,
                env_settings,
                yml_config_setting,
            )


if __name__ == "__main__":
    testdict = Settings().dict()

    print(testdict)
    print("Done")
