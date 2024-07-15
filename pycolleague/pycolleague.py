from __future__ import annotations

import collections.abc
import glob
import re
import urllib
from pathlib import Path
from typing import Union

import duckdb as ddb
import pandas as pd
import polars as pl
import sqlalchemy

if __name__ == "__main__":
    # Use this form when used locally.
    from config import Settings

else:
    # This form works when used as a package, but not locally.
    from pycolleague.config import Settings


class ColleagueError(Exception):
    pass


class ColleagueConfigurationError(Exception):
    pass


class ColleagueConnection(object):
    """
    Connection to CCDW Data Warehouse built from Colleague data extraction.
    """

    source: str = ""
    sourcepath: Union(str, None) = ""
    config: dict = {}
    config__sql__schema_history: str = ""
    conn_details: str = ""
    engine: Union(sqlalchemy.engine.base.Engine, ddb.DuckDBPyConnection) = None
    df_format: str = ""
    lazy: bool = False
    read_only: bool = True

    def __init__(
        self,
        *,
        source: str = "",
        sourcepath: str = "",
        format: str = "pandas",
        lazy: bool = False,
        config: dict = None,
        read_only: bool = None,
    ) -> None:
        """
        The constructor for ColleagueConnection class.

        Parameters:
            source (string):        Specify source of data. Can be ccdw, datamart, duckdb, or file.
            sourcepath (string):    Default="". Specify the path to the data source. For ccdw, this is
                                    blank. For datamart, this is the root folder for the datamart.
                                    For file, this is the path to the folder containing the files. For
                                    duckdb, this is the path to the duckdb file. This can also be specified
                                    in the config file under config:location_duckdb.
            format (string):        Default="pandas". Specify the format of the output. This can be
                                    pandas or polars. If pandas, the output will be a pandas DataFrame.
                                    If polars, the output will be a polars DataFrame.
            lazy (bool):            Default=False. Specify whether to load the data lazily or not. If True,
                                    the data will not be loaded until the user requests it. If False, the
                                    data will be loaded immediately.
            config (dictionary):    Default=None. Pass in the config file or the constructor will load the
                                    config file itself.
            read_only (bool):       Default=True. Specify whether the connection is read-only or not. If True,
                                    the connection will be read-only. If False, the connection will be read-write.
        """

        if format.lower() == "pandas":
            self.df_format = "pandas"
        elif format.lower() == "polars":
            self.df_format = "polars"
        else:
            # Raise error
            self.df_format = "pandas"

        if self.df_format == "pandas":
            self.lazy = False
        else:
            self.lazy = lazy

        if config:
            self.config = config.copy()
        else:
            self.config = Settings().model_dump()

        if source and source.lower() in ["ccdw", "datamart", "duckdb", "file"]:
            self.source = source.lower()
        elif "config" in self.config and "source" in self.config["config"]:
            self.source = self.config["config"]["source"]
        else:
            self.source = "file"

        if self.source not in ["ccdw", "datamart", "duckdb", "file"]:
            # Raise error
            self.source = "file"

        if read_only:
            self.read_only = read_only
        elif (
            self.source == "duckdb"
            and "duckdb" in self.config
            and "read_only" in self.config["duckdb"]
        ):
            self.read_only = self.config["duckdb"]["read_only"]
        elif "config" in self.config and "read_only" in self.config["config"]:
            self.read_only = self.config["config"]["read_only"]
        else:
            self.read_only = True

        if self.source == "ccdw":
            self.conn_details = urllib.parse.quote_plus(
                f"DRIVER={{{self.config['sql']['driver']}}};"
                f"SERVER={self.config['sql']['server']};"
                f"DATABASE={self.config['sql']['db']};"
                f"Trusted_Connection=Yes;"
                f"Description=Python ColleagueConnection Class"
            )
            if self.df_format == "pandas":
                self.sourcepath = f"mssql+pyodbc:///?odbc_connect={self.conn_details}"
                self.engine = sqlalchemy.create_engine(self.sourcepath)

            elif self.df_format == "polars":
                self.sourcepath = f"mssql://{self.config['sql']['server']}:1433/{self.config['sql']['db']}?trusted_connection=true"
                # self.engine = sqlalchemy.create_engine(self.sourcepath)
                # mssql://researchvm.haywood.edu:1433/CCDW_HIST?trusted_connection=true

            self.config__sql__schema_history = self.config["sql"]["schema_history"]

        elif self.source == "datamart":
            if sourcepath:
                self.sourcepath = sourcepath
            elif "datamart" in self.config and "rootfolder" in self.config["datamart"]:
                self.sourcepath = self.config["datamart"]["rootfolder"]
            elif (
                "pycolleague" in self.config
                and "sourcepath" in self.config["pycolleague"]
            ):
                self.sourcepath = self.config["pycolleague"]["sourcepath"]
            else:
                self.sourcepath = ""

        elif self.source == "duckdb":
            if sourcepath:
                self.sourcepath = sourcepath
            elif "duckdb" in self.config and "path" in self.config["duckdb"]:
                self.sourcepath = self.config["duckdb"]["path"]
            else:
                self.sourcepath = None

            if self.sourcepath:
                self.engine = ddb.connect(
                    database=self.sourcepath, read_only=self.read_only
                )
            else:
                # ERROR

                pass

        else:
            if sourcepath:
                self.sourcepath = sourcepath
            elif (
                "pycolleague" in self.config
                and "sourcepath" in self.config["pycolleague"]
            ):
                self.sourcepath = self.config["pycolleague"]["sourcepath"]
            else:
                self.sourcepath = "."

    def __get_data_ccdw__(
        self,
        colleaguefile: str,
        cols: Union[dict, list] = [],
        where: str = "",
        schema: str = "",
        version: str = "current",
        index_col: bool = False,
        # format: str = "pandas",
        debug: str = "",
    ) -> (pd.DataFrame, pl.DataFrame, pl.LazyFrame):
        if isinstance(cols, collections.abc.Mapping):
            qry_cols = (
                "*"
                if cols == []
                else ", ".join([f'"{c}" AS "{cols[c]}"' for c in cols])
            )

        else:
            qry_cols = "*" if cols == [] else ", ".join([f'"{c}"' for c in cols])
        qry_meta_where_cols = (
            "AND COLUMN_NAME IN (" + ", ".join([f"'{c}'" for c in cols]) + ")"
        )

        if where != "":
            qry_where_base = where

            # Convert all double-quotes (") to single-quotes (')
            # qry_where_base = qry_where_base.replace('"', "'")

            # Convert VAR IN ['ITEM1','ITEM2'] into VAR IN ('ITEM1','ITEM2'),
            #     or VAR NOT IN ['ITEM1','ITEM2'] into VAR NOT IN ('ITEM1','ITEM2')
            for fnd in re.findall(
                r"\s+(?:not\s+)?in\s+\[([^]]+)\]", qry_where_base, re.IGNORECASE
            ):
                qry_where_base = qry_where_base.replace(f"[{fnd}]", f"({fnd})")

            # Convert ['ITEM1'] into 'ITEM1'
            for fnd in re.findall(r"\[('.*?')\]", qry_where_base):
                qry_where_base = qry_where_base.replace(f"[{fnd}]", f"{fnd}")

            # Find any VAR.NAME type variables but also find [VAR.NAME].
            # Convert VAR.NAME and [VAR.NAME] into "VAR.NAME".
            # This will leave "VAR.NAME" alone.
            for fnd in re.findall(
                r"((?<!\")\[?\b[A-Za-z]+\.[A-Za-z\.]+\b\]?(?!\"))", qry_where_base
            ):
                qry_where_base = (
                    qry_where_base.replace(fnd, f'"{fnd[1:-1]}"')
                    if fnd[0] == "["
                    else qry_where_base.replace(fnd, f'"{fnd}"')
                )

                # qry_where_base = (
                #     qry_where_base
                #     if f[0] == "["
                #     else qry_where_base.replace(f, f"[{f}]")
                # )

            # Convert [VARNAME] into "VARNAME".
            for fnd in re.findall(r"(\[\w*\])", qry_where_base):
                qry_where_base = (
                    qry_where_base.replace(fnd, f'"{fnd[1:-1]}"')
                    if fnd[0] == "["
                    else qry_where_base.replace(fnd, f'"{fnd}"')
                )

            # Convert remaining == to =
            qry_where_base = qry_where_base.replace("==", "=")

            # Convert remaining != to <>
            qry_where_base = qry_where_base.replace("!=", "<>")

            qry_where = "" if where == "" else f"WHERE {qry_where_base}"
        else:
            qry_where = ""

        if schema == "":
            schema = self.config__sql__schema_history

        if version == "current" and schema == self.config__sql__schema_history:
            qry_where = "WHERE " if where == "" else qry_where + " AND "
            qry_where += f"CurrentFlag='Y'"

        qry = f"SELECT {qry_cols} FROM {schema}.{colleaguefile} {qry_where}"

        if debug == "query":
            print(qry)

        if schema.upper() != "INFORMATION_SCHEMA":
            if self.source == "duckdb":
                qry_meta = f"""
                    SELECT COLUMN_NAME AS "COLUMN_NAME"
                        , DATA_TYPE AS "DATA_TYPE"
                        , CASE WHEN DATA_TYPE IN ('FLOAT','FLOAT4','FLOAT8','REAL','NUMERIC','DECIMAL','DOUBLE') THEN 'float'
                                WHEN DATA_TYPE IN ('BLOB','BINARY','BYTEA','VARBINARY') THEN 'bytes'
                                WHEN DATA_TYPE IN ('BIT','BITSTRING','LOGICAL','BOOL') THEN 'bool'
                                WHEN DATA_TYPE IN ('DATE','DATETIME','TIME','DATETIME2','TIMESTAMP','TIMESTAMP WITH TIME ZONE','TIMESTAMPZ','TIMESTAMP_NS') THEN 'datetime'
                                WHEN DATA_TYPE IN ('BIGINT','INT1','INT2','INT4','INT8','LONG','HUGEINT','SIGNED','SHORT','SMALLINT','TINYINT','INT',
                                                'UBIGINT','UINTEGER','USMALLINT','UTINYINT') THEN 'int'
                                ELSE 'str' END AS PYTHON_DATA_TYPE
                    FROM duckdb_columns()
                    WHERE SCHEMA_NAME = '{schema}'
                    AND TABLE_NAME = '{colleaguefile}'
                    {qry_meta_where_cols}
                    """
            else:
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
                    {qry_meta_where_cols}
                    """

            # if self.source == "duckdb":
            #     df_meta = self.engine.sql(qry_meta).df()
            # elif self.source == "ccdw":
            # Check for polars
            #     df_meta = pd.read_sql(qry_meta, self.engine)
            # else:
            #     # Raise error
            #     df_meta = None
            # cache table column types

            # if isinstance(cols, collections.abc.Mapping):
            #     df_meta = df_meta.rename(cols)

            # df_types = df_meta[["COLUMN_NAME", "PYTHON_DATA_TYPE"]].to_dict()

        if self.source == "ccdw":
            if self.df_format == "pandas":
                df = pd.read_sql(qry, self.engine)  # , index_col = index_col)
            elif self.df_format == "polars":
                df = pl.read_database_uri(
                    qry, self.sourcepath
                )  # , index_col = index_col)
        elif self.source == "duckdb":
            if self.df_format == "pandas":
                df = self.engine.sql(qry).df()
            elif self.df_format == "polars":
                df = self.engine.sql(qry).pl()
        else:
            # Raise error
            df = None

        if self.lazy:
            df = df.lazy()
            # Need to apply the schema to the lazy frame
            # df = df.with_column_types(df_types["PYTHON_DATA_TYPE"])

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

        colleaguefile_pattern = Path(self.sourcepath) / f"{colleaguefile}*.csv"

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
        return self.config

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
        sql_sources = ("ccdw", "duckdb")
        if self.source in sql_sources:
            df = self.__get_data_ccdw__(
                colleaguefile,
                cols=cols,
                where=where,
                schema=schema,
                version=version,
                debug=debug,
            )
        # elif self.source == "datamart":
        #     df = self.__get_data_datamart__(colleaguefile, cols, where, debug)
        elif self.source == "file":
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
            sql_code:       The SQL code to execute. Must be connected to CCDW or DuckDB.
            debug (str):    Default="". Specify the debug level. Valid debug levels:
                            query: print out the generated query
        """
        newline = "\n"

        if debug == "query":
            print(f"SQL:{newline}{sql_code}")

        # TODO: Validate sql_code to ensure it is valid SQL code.

        if self.source == "ccdw":
            if self.df_format == "pandas":
                df = pd.read_sql(sql_code, self.engine)  # , index_col = index_col)
            elif self.df_format == "polars":
                df = pl.read_database_uri(
                    sql_code, self.sourcepath
                )  # , index_col = index_col)
        elif self.source == "duckdb":
            if self.df_format == "pandas":
                df = self.engine.sql(sql_code).df()
            elif self.df_format == "polars":
                df = self.engine.sql(sql_code).pl()
        else:
            # Raise error
            df = None

        if self.lazy:
            df = df.lazy()

        return df

    def School_ID(self):
        """Return the school's InstID from the config file."""
        return self.config["school"]["instid"]

    def School_IPEDS(self):
        """Return the school's IPEDS ID from the config file."""
        return self.config["school"]["ipeds"]


# For testing purposes only
if __name__ == "__main__":
    report_terms = ["2020FA", "2020SP"]

    testsource = "duckdb"

    ddb_conn_pl = ColleagueConnection(
        source=testsource, sourcepath="duckdb_test.db", format="polars", lazy=True
    )
    df = ddb_conn_pl.get_data(
        "Term_CU",
        schema="dw_dim",
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
    print(df.explain)
    df2 = df.collect()
    print(df2)

    df = ddb_conn_pl.get_data(
        "Term_CU",
        schema="dw_dim",
        cols={
            "Term_ID": "Term",
            "Academic_Year": "AY",
            "Reporting_Year": "RY",
            "Term_Start_Date": "Start",
            "Term_End_Date": "End",
        },
        where=f"[Term_ID] IN {report_terms}",
        debug="query",
    )
    print(df.explain)
    df2 = df.collect()
    print(df2)

    testsource: str = "ccdw"

    ccdw_conn = ColleagueConnection(source=testsource)

    analytics_where = "AND TABLE_NAME IN ('Term_CU','Dates')"
    tbls = ccdw_conn.get_data(
        "TABLES",
        schema="INFORMATION_SCHEMA",
        version="full",
        cols=["TABLE_SCHEMA", "TABLE_NAME"],
        where=f"""
            "TABLE_TYPE"='BASE TABLE' 
            AND [TABLE_SCHEMA] IN ('dw_dim','history','local','public','main')
            /*AND TABLE_NAME = 'XCOURSE_SECTIONS'*/
            {analytics_where}
        """,
        debug="query",
    )
    print(tbls)

    print(
        ccdw_conn.get_data(
            "ACAD_PROGRAMS",
            cols=[
                "ACAD.PROGRAMS.ID",
                "ACPG.ACAD.LEVEL",
            ],
            where="""
                [ACAD.PROGRAMS.ID] IN ['A10100','A10100EC']
                AND "ACPG.ACAD.LEVEL" IN ['CU']
            """,
            debug="query",
        )
    )
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

    ccdw_conn_pl = ColleagueConnection(source=testsource, format="polars")
    print(
        ccdw_conn_pl.get_data(
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

    ccdw_conn_pll = ColleagueConnection(source=testsource, format="polars", lazy=True)
    df = ccdw_conn_pll.get_data(
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
    print(df.explain)
    df2 = df.collect()
    print(df2)

    print(
        f"ddb_conn_pl: source={ddb_conn_pl.source}, sourcepath={ddb_conn_pl.sourcepath}, df_format={ddb_conn_pl.df_format}, lazy={ddb_conn_pl.lazy}"
    )
    print(
        f"ccdw_conn: source={ccdw_conn.source}, sourcepath={ccdw_conn.sourcepath}, df_format={ccdw_conn.df_format}, lazy={ccdw_conn.lazy}"
    )
    print(
        f"ccdw_conn_pl: source={ccdw_conn_pl.source}, sourcepath={ccdw_conn_pl.sourcepath}, df_format={ccdw_conn_pl.df_format}, lazy={ccdw_conn_pl.lazy}"
    )
    print(
        f"ccdw_conn_pll: source={ccdw_conn_pll.source}, sourcepath={ccdw_conn_pll.sourcepath}, df_format={ccdw_conn_pll.df_format}, lazy={ccdw_conn_pll.lazy}"
    )

    print("done")
