import multiprocessing

bind = "0.0.0.0:10000"

workers = 1                 # ⚠️ MUST be 1 (SQLite + Face models unsafe with multi workers)
threads = 4                 # Parallel requests inside same worker

timeout = 180               # Heavy face model needs time
graceful_timeout = 180
keepalive = 5

preload_app = True          # Load model once at boot (huge speed boost)
max_requests = 200
max_requests_jitter = 50

loglevel = "info"
accesslog = "-"
errorlog = "-"
