import os


def get_non_empty_value(env_var_name: str, default_value: any = None) -> any:
    """ Get the non empty value from the environment variable """
    resolved_value = os.environ.get(env_var_name)
    if resolved_value is None or resolved_value == "":
        resolved_value = default_value
    return resolved_value
