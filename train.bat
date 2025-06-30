@echo off
for /L %%i in (1,1,45) do (
    python Simulate.py %%i
    python Learn.py %%i
)