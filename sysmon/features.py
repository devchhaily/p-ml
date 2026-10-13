"""Single source of truth for which metrics the model sees, and in what order."""

FEATURES: list[str] = [
    "cpu_percent",      # 0-100, all cores averaged
    "mem_percent",      # 0-100, used RAM
    "swap_percent",     # 0-100
    "load_per_cpu",     # 1-min load average / logical CPU count
    "num_procs",        # running process count
    "disk_read_mbs",    # MB/s
    "disk_write_mbs",   # MB/s
    "net_sent_mbs",     # MB/s
    "net_recv_mbs",     # MB/s
]

# Throughput metrics are heavy-tailed (idle ~0, bursts in the hundreds),
# so they are log1p-compressed before standardisation.
LOG_FEATURES: set[str] = {"disk_read_mbs", "disk_write_mbs", "net_sent_mbs", "net_recv_mbs"}
