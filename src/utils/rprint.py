# coding: utf-8

"""
custom print and log functions 
"""

from rich import print

__all__ = ['rprint', 'rlog']

rprint = print
rlog = lambda msg: print(f"[cyan]{time.ctime().split()[3]}[/cyan] [green]INFO[/green] - {msg}")
