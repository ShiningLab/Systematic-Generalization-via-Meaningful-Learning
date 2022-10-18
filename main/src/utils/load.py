#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


# built-in
import json


def load_json(path: str) -> list:
	with open(path, 'r', encoding='utf-8') as f:
		return json.load(f)