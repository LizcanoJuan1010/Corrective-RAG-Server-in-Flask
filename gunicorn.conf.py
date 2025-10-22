# Gunicorn config file
# https://docs.gunicorn.org/en/stable/settings.html

# The address to bind to.
bind = "0.0.0.0:8000"

# The number of worker processes.
# This should be (2 x $num_cores) + 1
workers = 5

# The type of worker to use.
worker_class = "sync"

# The number of threads to use.
threads = 1

# The maximum number of requests a worker will process before restarting.
max_requests = 1000

# The maximum number of requests a worker will process before restarting.
max_requests_jitter = 50

# The timeout for workers.
timeout = 30

# The log level.
loglevel = "info"
