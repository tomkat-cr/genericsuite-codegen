"""
Settings manager for GenericSuite Codegen.

Configuration editor that reads the variables and labels from the
".env.example" file.
The labels are the comment lines (the ones begining with "#" and aren't
followed by a uppercase environment variable name and a "=" sign).
The input texts are the uppercase environment variable name followed by a "="
sign.
The values are saved in the "main_config.json" file.
If the "main_config.json" file does not exist, it will be created from the
"main_config.template.json" file.
If the "main_config.template.json" file does not exist, it will be created
from the ".env" file.
"""

import os
import json
import re
from typing import Dict, Any, List

from genericsuite_codegen.api.types import (
    SettingItem,
    SettingItemType,
    # SettingsResponse,
    UpdateSettingsRequest,
)

from genericsuite_codegen.utilities import (
    std_error_response,
    std_response,
)
from genericsuite_codegen.utilities.app_logger import (
    log_debug,
    log_error,
)


DEBUG = False


class SettingsManager:
    """
    Settings manager for GenericSuite Codegen.
    """

    def __init__(self):
        self.exclusions = [r"^APP_DB_URI$", r"^HF_TOKEN$", r".*_API_KEY$"]

    async def get_all(self, filter_sensitive_data: bool = True
                      ) -> Dict[str, Any]:
        """
        Get application settings based on .env.example.

        Returns:
            Dict[str, Any]: Standard response with SettingsResponse.
        """
        try:
            settings: List[SettingItem] = []

            # Paths
            # Project root is 3 levels up from this file
            project_root = os.path.abspath(os.path.join(
                os.path.dirname(__file__), "..", "..", ".."))

            env_example_path = os.path.join(project_root, ".env.example")

            assets_dir = os.path.join(
                project_root, "server", "genericsuite_codegen", "assets")

            main_json_path = os.path.join(assets_dir, "main_config.json")

            template_json_path = os.path.join(assets_dir,
                                              "main_config.template.json")

            env_path = os.path.join(project_root, ".env")

            # Load current values
            current_values = {}

            # 1. Start with .env
            if os.path.exists(env_path):
                with open(env_path, "r", encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, val = line.split("=", 1)
                            current_values[key.strip()] = val.strip().strip(
                                '"').strip("'")

            # 2. Override with main_config.template.json
            if os.path.exists(template_json_path):
                try:
                    with open(template_json_path, "r", encoding='utf-8') as f:
                        template_data = json.load(f)
                        current_values.update(template_data)
                except Exception as e:
                    log_error(f"Error loading {template_json_path}: {e}")

            # 3. Override with main_config.json
            if os.path.exists(main_json_path):
                try:
                    with open(main_json_path, "r", encoding='utf-8') as f:
                        main_data = json.load(f)
                        current_values.update(main_data)
                except Exception as e:
                    log_error(f"Error loading {main_json_path}: {e}")

            # Check if .env.example exists
            if not os.path.exists(env_example_path):
                return std_error_response(
                    status_code=404,
                    detail=".env.example not found"
                )

            # Parse .env.example
            with open(env_example_path, "r", encoding='utf-8') as f:
                last_variable = None
                last_original_value = None
                for line in f:
                    orig_line = line.strip()
                    if not orig_line:
                        continue

                    # Check for variables: UPPERCASE_NAME=...
                    var_match = re.match(r"^([A-Z0-9_]+)=(.*)", orig_line)
                    if var_match:
                        var_name = var_match.group(1)
                        last_original_value = var_match.group(2)
                        if filter_sensitive_data and any(
                                re.fullmatch(pattern, var_name)
                                for pattern in self.exclusions):
                            # Exclude sensitive data, to avoid transmit it
                            # over the network. E.g. database credentials,
                            # api keys, etc. These variables can only be
                            # changed in the .env file
                            continue

                        settings.append(SettingItem(
                            type=SettingItemType.VARIABLE,
                            name=var_name,
                            label=var_name,
                            value=str(current_values.get(var_name, ""))
                        ))

                        last_variable = var_name
                        continue

                    # Check for labels: # ... but not # VARIABLE=...
                    if orig_line.startswith("#"):
                        # Remove leading # and spaces
                        label_text = orig_line[1:].strip()
                        # Check if it looks like a commented out variable
                        var_match = re.match(r"^([A-Z0-9_]+)=", label_text)
                        if not var_match:
                            settings.append(SettingItem(
                                type=SettingItemType.LABEL,
                                label=label_text
                            ))
                        else:
                            var_name = var_match.group(1)
                            if last_variable == var_name:
                                # If the last item is a variable, and this one
                                # the same variable commented out, the
                                # variable is a select option
                                if settings[-1].select_options is None:
                                    # Include the current value as an option
                                    settings[-1].select_options = \
                                        [last_original_value]
                                value = label_text.split("=")[1].strip()
                                settings[-1].select_options.append(value)

            _ = DEBUG and log_debug(
                f"Loaded settings: {settings}")

            return std_response(result={
                "settings": [item.model_dump() for item in settings]
            })

        except Exception as e:
            log_error(f"Failed to get settings: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to get settings: {str(e)}"
            )

    async def update(
        self,
        request: UpdateSettingsRequest
    ) -> Dict[str, Any]:
        """
        Update application settings.

        Args:
            request: Update settings request.

        Returns:
            Dict[str, Any]: Standard response.
        """
        try:
            # Project root is 3 levels up from this file
            project_root = os.path.abspath(os.path.join(
                os.path.dirname(__file__), "..", "..", ".."))

            main_json_path = os.path.join(
                project_root, "server", "genericsuite_codegen", "assets",
                "main_config.json")

            # Ensure directory exists
            os.makedirs(os.path.dirname(main_json_path), exist_ok=True)

            # Load existing if any
            existing_settings = await self.get_all(filter_sensitive_data=False)
            if existing_settings.error:
                return existing_settings

            _ = DEBUG and log_debug(
                f"Request settings: {request.settings}")

            existing_data_list = existing_settings.result["settings"]

            _ = DEBUG and log_debug(
                f"Existing settings: {existing_data_list}")

            # existing_data = {}
            # if os.path.exists(main_json_path):
            #     try:
            #         with open(main_json_path, "r", encoding='utf-8') as f:
            #             existing_data = json.load(f)
            #     except Exception:
            #         pass

            # Update with new values
            # Assign "" to "_blank_option_" values
            existing_data = {}
            for item in existing_data_list:
                _ = DEBUG and log_debug(
                    f"Item: {item}")

                if not item or not item["name"] or item["name"] == "null":
                    continue
                new_value = item["value"]
                if request.settings.get(item["name"]):
                    new_value = request.settings[item["name"]]
                elif os.getenv(item["name"]):
                    new_value = os.getenv(item["name"])
                if new_value == "_blank_option_":
                    new_value = ""
                existing_data[item["name"]] = str(new_value)

                _ = DEBUG and log_debug(
                    f"Item: {item['name']} | value: {new_value}")

            _ = DEBUG and log_debug(
                f"Data to be saved: {existing_data}")

            # Save to main_config.json
            with open(main_json_path, "w", encoding='utf-8') as f:
                json.dump(existing_data, f, indent=4)

            return std_response(result={
                "message": "Settings updated successfully"})

        except Exception as e:
            log_error(f"Failed to update settings: {e}")
            return std_error_response(
                status_code=500,
                detail=f"Failed to update settings: {str(e)}"
            )
