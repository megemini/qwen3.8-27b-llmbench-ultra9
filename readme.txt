Device name	NucBox_EVO-T1
Processor	Intel(R) Core(TM) Ultra 9 285H   2.90 GHz
Installed RAM	64.0 GB (63.5 GB usable)
Device ID	61D40F5E-4F5E-43A9-A90A-D9C2465F2792
Product ID	00355-61078-53024-AAOEM
System type	64-bit operating system, x64-based processor
Pen and touch	No pen or touch input is available for this display

GPU 		Inter(R) Arc(TM) 140T GPU (32GB)

Edition	Windows 11 Pro
Version	24H2
Installed on	‎8/‎26/‎2026
OS build	26100.3476
Experience	Windows Feature Experience Pack 1000.26100.54.0

Herdsman
llmbench -u http://localhost:8080/v1 --runs 3 -m Qwen3.8-27B

unsloth
llmbench -u http://127.0.0.1:8888/v1 --runs 3 -m Qwen3.8-27B-GGUF:Q4_K_M --api-key sk-unsloth-27d4f789420eae7ab7f89cba0dd0173f

Bionic
llmbench -u http://localhost:1234/v1 --runs 3 -m qwen3.8-27b

ovms
llmbench -u http://localhost:8000/v1 --runs 3
