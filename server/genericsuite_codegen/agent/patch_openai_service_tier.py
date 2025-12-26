"""
Monkey patch OpenAI ChatCompletion to accept 'on_demand' service_tier.
Reference:
    https://github.com/pydantic/pydantic-ai/issues/2499
Error that occurred:
    Failed to process query: Invalid response from OpenAI chat completions
    endpoint: 1 validation error for ChatCompletion
    service_tier
       Input should be 'auto', 'default', 'flex', 'scale' or 'priority'
       [type=literal_error, input_value='on_demand', input_type=str]
         For further information visit
           https://errors.pydantic.dev/2.11/v/literal_error
"""
from typing_extensions import Literal
from typing import Optional
from openai.types.chat.chat_completion import ChatCompletion
import pydantic


def patch_openai_service_tier() -> None:
    """Monkey patch OpenAI ChatCompletion to accept 'on_demand' service_tier.

    The OpenAI API specification allows additional service_tier values beyond
    the ones defined in the Python client. This patch extends the validation
    to include 'on_demand' which is used by some compatible providers.
    """
    # Get the current field info for service_tier
    current_field = ChatCompletion.model_fields.get('service_tier')
    if current_field is None:
        return

    # Create new field info with extended Literal types
    extended_annotation = Optional[Literal[
        "auto", "default", "flex", "scale", "priority", "on_demand"]]

    # Update the field annotation
    new_field = pydantic.fields.FieldInfo(
        annotation=extended_annotation,
        default=current_field.default,
        description=current_field.description,
    )

    # Update the model fields and rebuild
    ChatCompletion.model_fields['service_tier'] = new_field
    ChatCompletion.model_rebuild()
