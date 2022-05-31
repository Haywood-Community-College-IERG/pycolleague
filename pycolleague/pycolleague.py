from __future__ import annotations

import collections.abc
import glob
import os
import re
import sys
import urllib
from pathlib import Path
from typing import Union

import config as cfg
import pandas as pd
import sqlalchemy


class ColleagueError(Exception):
    pass


class ColleagueConfigurationError(Exception):
    pass


class ColleagueConnection(object):
    """
    Connection to CCDW Data Warehouse built from Colleague data extraction.
    """

    def __init__(
        self, source: str = "", sourcepath: str = "", config: dict = None
    ) -> None:
        """
        The constructor for ColleagueConnection class.

        Parameters:
            source (string):        Specify source of data. Can be ccdw, datamart, or file.
            config (dictionary):    Default=None. Pass in the config file or the constructor will load the
                                    config file itself.
        """
        if config:
            self.__config__ = config.copy()
        else:
            self.__config__ = cfg.Settings().dict()

        if source:
            self.__source__ = source.lower()
        elif (
            "pycolleague" in self.__config__
            and "source" in self.__config__["pycolleague"]
        ):
            self.__source__ = self.__config__["pycolleague"]["source"]
        else:
            self.__source__ = "file"

        if self.__source__ not in ["ccdw", "datamart", "file"]:
            # Raise error
            pass

        if self.__source__ == "ccdw":
            self.__conn_details__ = urllib.parse.quote_plus(
                f"DRIVER={{{self.__config__['sql']['driver']}}};"
                f"SERVER={self.__config__['sql']['server']};"
                f"DATABASE={self.__config__['sql']['db']};"
                f"Trusted_Connection=Yes;"
                f"Description=Python ColleagueConnection Class"
            )
            self.__sourcepath__ = (
                f"mssql+pyodbc:///?odbc_connect={self.__conn_details__}"
            )
            self.__engine__ = sqlalchemy.create_engine(self.__sourcepath__)

            self.__config_schema_history__ = self.__config__["sql"]["schema_history"]

        elif self.__source__ == "datamart":
            if sourcepath:
                self.__sourcepath__ = sourcepath
            elif (
                "datamart" in self.__config__
                and "rootfolder" in self.__config__["datamart"]
            ):
                self.__sourcepath__ = self.__config__["datamart"]["rootfolder"]
            elif (
                "pycolleague" in self.__config__
                and "sourcepath" in self.__config__["pycolleague"]
            ):
                self.__sourcepath__ = self.__config__["pycolleague"]["sourcepath"]
            else:
                self.__sourcepath__ = "."

        else:
            if sourcepath:
                self.__sourcepath__ = sourcepath
            elif (
                "pycolleague" in self.__config__
                and "sourcepath" in self.__config__["pycolleague"]
            ):
                self.__sourcepath__ = self.__config__["pycolleague"]["sourcepath"]
            else:
                self.__sourcepath__ = "."

    def __get_data_ccdw__(
        self,
        colleaguefile: str,
        cols: Union[dict, list] = [],
        where: str = "",
        schema: str = "history",
        version: str = "current",
        index_col: bool = False,
        debug: str = "",
    ):
        if isinstance(cols, collections.abc.Mapping):
            qry_cols = (
                "*"
                if cols == []
                else ", ".join([f"[{c}] AS [{cols[c]}]" for c in cols])
            )
        else:
            qry_cols = "*" if cols == [] else ", ".join([f"[{c}]" for c in cols])

        # qry_meta_where_cols = '' if cols == [] else "AND COLUMN_NAME IN (" & ', '.join([f"'{c}'" for c in cols]) & ')'

        if where != "":
            qry_where_base = where

            # Convert VAR IN ['ITEM1','ITEM2'] into VAR IN ('ITEM1','ITEM2'),
            #     or VAR NOT IN ['ITEM1','ITEM2'] into VAR NOT IN ('ITEM1','ITEM2')
            for f in re.findall(
                r"\s+(?:not\s+)?in\s+\[([^]]+)\]", qry_where_base, re.IGNORECASE
            ):
                qry_where_base = qry_where_base.replace(f"[{f}]", f"({f})")

            # Convert ['ITEM1'] into 'ITEM1'
            for f in re.findall(r"\[('.*?')\]", qry_where_base):
                qry_where_base = qry_where_base.replace(f"[{f}]", f"{f}")

            # Find any VAR.NAME type variables but also find [VAR.NAME].
            # Convert VAR.NAME into [VAR.NAME] and leave the others alone.
            for f in re.findall(r"(\[?\w*\.[\w\.]+\]?)", qry_where_base):
                qry_where_base = (
                    qry_where_base
                    if f[0] == "["
                    else qry_where_base.replace(f, f"[{f}]")
                )

            # Convert all double-quotes (") to single-quotes (')
            qry_where_base = qry_where_base.replace('"', '"')

            # Convert remaining == to =
            qry_where_base = qry_where_base.replace("==", "=")

            # Convert remaining != to <>
            qry_where_base = qry_where_base.replace("!=", "<>")

            qry_where = "" if where == "" else f"WHERE {qry_where_base}"
        else:
            qry_where = ""

        if version == "current" and schema == self.__config_schema_history__:
            qry_where = "WHERE " if where == "" else qry_where + " AND "
            qry_where += f"CurrentFlag='Y'"

        qry = f"SELECT {qry_cols} FROM {schema}.{colleaguefile} {qry_where}"

        if debug == "query":
            print(qry)

        # Check for existence in global cache
        # if self.__cache__[f"{schema}.{colleaguefile}"]:
        #     df_meta = self.__cache__[f"{schema}.{colleaguefile}"]
        # else:
        qry_meta = f"""
            SELECT TABLE_SCHEMA
                 , TABLE_NAME
                 , COLUMN_NAME
                 , DATA_TYPE
                 , CASE WHEN DATA_TYPE IN ('bigint','float','real','numeric','decimal','money','smallmoney') THEN 'float'
                        WHEN DATA_TYPE IN ('binary','varbinary','image') THEN 'bytes'
                        WHEN DATA_TYPE IN ('bit') THEN 'bool'
                        WHEN DATA_TYPE IN ('date','datetime','smalldatatime','time','datetime2') THEN 'datatime'
                        WHEN DATA_TYPE IN ('smallint','tinyint','int') THEN 'int'
                        ELSE 'str' END AS PYTHON_DATA_TYPE 
              FROM INFORMATION_SCHEMA.COLUMNS
             WHERE TABLE_SCHEMA = '{schema}'
               AND TABLE_NAME = '{colleaguefile}'
            """
        df_meta = pd.read_sql(qry_meta, self.__engine__)
        # cache table column types

        if isinstance(cols, collections.abc.Mapping):
            df_meta = df_meta.rename(cols)

        df_types = df_meta[["COLUMN_NAME", "PYTHON_DATA_TYPE"]].to_dict()

        df = pd.read_sql(qry, self.__engine__)  # , index_col = index_col)

        return df

    def __dictsub(self, text, kw, ignore_case=False):
        search_keys = map(lambda x: re.escape(x), kw.keys())
        if ignore_case:
            kw = {k.lower(): kw[k] for k in kw}
            regex = re.compile("|".join(search_keys), re.IGNORECASE)
            res = regex.sub(lambda m: kw[m.group().lower()], text)
        else:
            regex = re.compile("|".join(search_keys))
            res = regex.sub(lambda m: kw[m.group()], text)

        return res

    def __get_data_datamart__(
        self,
        colleaguefile: str,
        cols: Union[dict, list] = [],
        where: str = "",
        index_col: bool = False,
        debug: str = "",
    ):
        pass

    def __get_data_file__(
        self,
        colleaguefile: str,
        cols: Union[dict, list] = [],
        where: str = "",
        index_col: bool = False,
        debug: str = "",
    ):
        if isinstance(cols, collections.abc.Mapping):
            qry_cols = "*" if cols == [] else [c for c in cols]
            qry_cols_equiv = (
                "*"
                if cols == []
                else ", ".join([f"[{c}] AS [{cols[c]}]" for c in cols])
            )
        else:
            qry_cols = "*" if cols == [] else cols
            qry_cols_equiv = "*" if cols == [] else ", ".join([f"[{c}]" for c in cols])

        colleaguefile_pattern = Path(self.__sourcepath__) / f"{colleaguefile}*.csv"

        qry_where = where

        df = pd.DataFrame()
        for file in glob.glob(colleaguefile_pattern.__str__()):

            fdf = pd.read_csv(
                file,
                encoding="ansi",
                dtype="str",
                na_values=None,
                keep_default_na=False,
                engine="python",
            )
            fdf = fdf.where(pd.notnull(fdf), other=None)  # Keep only non-empty rows

            if df.empty:
                df = fdf
            else:
                df = df.append(fdf)

        where_col_correction = dict(
            zip(["[" + elem + "]" for elem in df.columns], df.columns)
        )
        qry_where = self.__dictsub(qry_where, where_col_correction, ignore_case=True)

        if debug == "query":
            print(
                f"Equivalent SQL: SELECT {qry_cols_equiv} FROM {colleaguefile} WHERE {qry_where}"
            )

        if qry_where:
            df = df.query(qry_where).reset_index(drop=True)

        if qry_cols != "*":
            df = df[qry_cols]

        if isinstance(cols, collections.abc.Mapping):
            df = df.rename(cols)

        return df

    def get_config(self):
        return self.__config__

    def get_data(
        self,
        colleaguefile: str,
        cols: Union[dict, list] = [],
        where: str = "",
        sep: str = ".",
        schema: str = "history",
        version: str = "current",
        # index_col: bool = False,
        debug: str = "",
    ):
        """
        Get data from Colleague data warehouse.

        Parameters:
            colleaguefile (str):    The base name of the Colleague file.
            cols (list or dict):    The list of columns to return from the specified file. You can specify
                                    new column names by using a dictionary.
            where (str):            All filters to be applied to the resulting table. These will be sent directly
                                    to SQL Server, but only basic Python filtering using IN, AND, OR, ==, != are allowed.
                                    All where conditions are applied before the columns are renamed.
            sep (str):              Default=".". Specify the separator value for column names. Colleague names are
                                    separated by '.'. Specifying a value here would replace that value with that
                                    character.
            version (str):          Default="current". Which version of the data to get. Options are
                                    "current" (default), "history", or "all". Option "current" adds
                                    "CurrentFlag='Y'" to the where argument. The other two are treated
                                    the same. This argument is ignored for non-SQL Server-based objects.
            schema (str):           Default="history". The schema from which to get the data. This argument is
                                    ignored for non-SQL Server-based objects.
            debug (str):            Default="". Specify the debug level. Valid debug levels:
                                    query: print out the generated query
        """
        if self.__source__ == "ccdw":
            df = self.__get_data_ccdw__(
                colleaguefile,
                cols=cols,
                where=where,
                schema=schema,
                version=version,
                debug=debug,
            )
        # elif self.__source__ == "datamart":
        #     df = self.__get_data_datamart__(colleaguefile, cols, where, debug)
        elif self.__source__ == "file":
            df = self.__get_data_file__(
                colleaguefile, cols=cols, where=where, debug=debug
            )
        else:
            # Raise error
            return None

        if sep != ".":
            df.columns = df.columns.str.replace(".", sep)

        return df

    def get_data_sql(self, sql_code: str, debug: str = ""):
        """
        Get data from Colleague data warehouse.

        Parameters:
            sql_code:       The SQL code to execute. Must be connected to CCDW.
            debug (str):    Default="". Specify the debug level. Valid debug levels:
                            query: print out the generated query
        """
        newline = "\n"
        if self.__source__ == "ccdw":
            if debug == "query":
                print(f"SQL:{newline}{sql_code}")
            return self.__engine__.execute(sql_code)
        else:
            pass

    def School_ID(self):
        """Return the school's InstID from the config file."""
        return self.__config__["school"]["instid"]

    def School_IPEDS(self):
        """Return the school's IPEDS ID from the config file."""
        return self.__config__["school"]["ipeds"]


# For testing purposes only
if __name__ == "__main__":
    testsource: str = "ccdw"

    report_terms = ["2020FA", "2020SP"]

    ccdw_conn = ColleagueConnection(source=testsource)
    print(
        ccdw_conn.get_data(
            "Term_CU",
            schema="dw_dim",
            #                version="all",
            cols=[
                "Term_ID",
                "Academic_Year",
                "Reporting_Year",
                "Term_Start_Date",
                "Term_End_Date",
            ],
            where=f"[Term_ID] IN {report_terms}",
            debug="query",
        )
    )
    print(
        ccdw_conn.get_data(
            "Term_CU",
            schema="dw_dim",
            version="all",
            cols=[
                "Term_ID",
                "Academic_Year",
                "Reporting_Year",
                "Term_Start_Date",
                "Term_End_Date",
            ],
            where=f"[Term_ID] in {report_terms} and [Reporting_Year] == '2020'",
            debug="query",
        )
    )
    print(
        ccdw_conn.get_data(
            "Term_CU",
            schema="dw_dim",
            version="all",
            cols={
                "Term_ID": "Term",
                "Academic_Year": "AY",
                "Reporting_Year": "RY",
                "Term_Start_Date": "Start",
                "Term_End_Date": "End",
            },
            where=f"Term_ID in {report_terms} and [Reporting_Year] == ['2020']",
            debug="query",
        )
    )
