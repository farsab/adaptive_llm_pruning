
def linear_schedule(step, total_steps, start=0.0, end=0.5):
    return start + (end - start) * (step / total_steps)
