from enum import Enum
import requests
import PyPDF2
import csv
import json
import pandas as pd


class Format(Enum):
    TXT = 1
    PDF = 2
    CSV = 3
    JSON = 4
    XLSX = 5


class ReadData:

    def read_data_from_url(self, url, data_type):
        response = requests.get(url)

        if data_type == Format.TXT:
            return response.text
        elif data_type == Format.PDF:
            pdf_reader = PyPDF2.PdfFileReader(response.content)
            text = ""
            for page_num in range(pdf_reader.numPages):
                page = pdf_reader.getPage(page_num)
                text += page.extract_text()
            return text
        elif data_type == Format.CSV:
            text = response.text
            data = csv.reader(text.splitlines())
            return list(data)
        elif data_type == Format.JSON:
            data = json.loads(response.text)
            return data
        elif data_type == Format.XLSX:
            with pd.ExcelFile(response.content) as xls:
                data = {}
                for sheet_name in xls.sheet_names:
                    data[sheet_name] = pd.read_excel(xls, sheet_name).values.tolist()
                return data
        else:
            return None

    def read_data_from_local_path(self, file_path, data_type):
        if data_type == Format.TXT:
            data = pd.read_csv(file_path, sep='\t', header=None).values.tolist()
            return data
        elif data_type == Format.PDF:
            pdf_reader = PyPDF2.PdfFileReader(file_path)
            text = ""
            for page_num in range(pdf_reader.numPages):
                page = pdf_reader.getPage(page_num)
                text += page.extract_text()
            return text
        elif data_type == Format.CSV:
            data = pd.read_csv(file_path)
            return data
        elif data_type == Format.JSON:
            data = pd.read_json(file_path)
            return data
        elif data_type == Format.XLSX:
            data = pd.read_excel(file_path, sheet_name=None)
            return data
        else:
            return None


# Example usage
url = "https://www.gutenberg.org/files/71037/71037-0.txt"
data_type = Format.TXT
r = ReadData()
data = r.read_data_from_url(url, data_type)
# print(data)
