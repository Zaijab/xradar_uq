# Real time to CR3BP time conversion
TU_seconds = 375730  # Time unit in seconds
TU_days = TU_seconds / (24 * 3600)  # ≈ 4.35 days

def real_time_to_cr3bp(real_seconds):
    return real_seconds / TU_seconds

def cr3bp_time_to_real(cr3bp_time):
    return cr3bp_time * TU_seconds
