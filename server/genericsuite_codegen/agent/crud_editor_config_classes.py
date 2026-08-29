"""
crud_editor_config_classes.py
2025-12-26 | CR

This file defines the Python classes to validate the frontend and backend JSON
configurations for the Generic CRUD Editor.
"""
from __future__ import annotations
from typing import List, Optional, Dict, Union, Literal
from pydantic import BaseModel, Field


# Defined Field Types for reference, though the interface allows any string
FieldType = Literal[
    'text', 'textarea', 'number', 'integer', 'date', 'datetime-local',
    'array', 'email', 'label', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'hr', '_id', 'select', 'select_table', 'select_component',
    'suggestion_dropdown', 'component'
]


class ParentKeyName(BaseModel):
    """
    Definition of a parent key relationship
    """
    parameterName: str = Field(
        ...,
        description="The name of the parameter to be used in the backend"
        " API call to identify the parent table Primary Key"
    )
    parentElementName: str = Field(
        ...,
        description="The name of the parent table's primary key "
        "column/attribute"
    )


class FieldElement(BaseModel):
    """
    Individual field configuration for the CRUD editor
    """
    name: str = Field(..., description="The name of the field element")
    label: str = Field(..., description="The label displayed for the field")
    type: Union[FieldType,
                str] = Field(..., description="The data type of the field")
    required: Optional[bool] = Field(
        None, description="Whether the field is required for submission")
    listing: Optional[bool] = Field(
        None, description="Whether the field should be displayed in the"
        " listing page")
    readonly: Optional[bool] = Field(
        None, description="Whether the field should be read-only")
    default_value: Optional[Union[str, int, float]] = Field(
        None, description="The default value to use for the field")
    hidden: Optional[bool] = Field(
        None, description="Whether the field should be hidden from display")
    formula: Optional[str] = Field(
        None, description="Formula function name to calculate value from"
        " other fields")
    # Select specific
    select_elements: Optional[str] = Field(
        None,
        description="Select specific: ID for predefined select options"
        " (e.g., 'TRUE_FALSE')"
    )
    # Suggestion Dropdown specific attributes
    suggestion_id_fieldname: Optional[str] = Field(
        None,
        description="Field name in API response for the selected item Key")
    suggestion_desc_fieldname: Optional[str] = Field(
        None,
        description="Field name in API response for the selected item Name")
    filter_api_url: Optional[str] = Field(
        None, description="API URL for suggestions")
    filter_api_request_method: Optional[str] = Field(
        None, description="HTTP method for suggestion API")
    filter_search_param_name: Optional[str] = Field(
        None, description="Parameter name for the search term")
    filter_search_other_param: Optional[Dict[str, str]] = Field(
        None, description="Additional parameters for the suggestion API")
    autocomplete_fields: Optional[Dict[str, str]] = Field(
        None, description="Fields to populate from the selected item")
    # Special Button attributes
    chatbot_popup: Optional[bool] = Field(
        None, description="Enable ChatBot button")
    chatbot_prompt: Optional[str] = Field(
        None,
        description="Text prompt for ChatBot button (%s replaced by"
        " field value)")
    google_popup: Optional[bool] = Field(
        None, description="Enable Google Search button")
    google_prompt: Optional[str] = Field(
        None,
        description="Text prompt for Google Search button (%s replaced by"
        " field value)")
    aux_component: Optional[str] = Field(
        None,
        description="Component name for auxiliary button (e.g. ChatBotButton)")
    uuid_generator: Optional[bool] = Field(
        None, description="Enable UUID generation for this field")

    # Recursive field
    fields: Optional[List[FieldElement]] = Field(
        None, description="Optional nested fields for array types")


class FrontendCrudEditorConfig(BaseModel):
    """
    Configuration for the Generic CRUD Editor (Frontend)
    """
    baseUrl: Optional[str] = Field(
        None, description="The ReactJS router URL for the CRUD editor")
    title: Optional[str] = Field(
        None, description="The title of the CRUD editor, used in the"
        " listing page")
    name: Optional[str] = Field(
        None,
        description="The name of the CRUD editor component, used as a title"
        " in the form submission page"
    )
    component: Optional[str] = Field(
        None, description="The ReactJS component name used by the CRUD editor")
    dbApiUrl: Optional[str] = Field(
        None, description="The API endpoint URL that the CRUD editor"
        " interacts with")
    primaryKeyName: Optional[str] = Field(
        "id",
        description="Primary Key parameter name for API calls (default: id)"
    )
    defaultOrder: Optional[str] = Field(
        None,
        description="The default order for sorting data in the listing page"
        " (e.g., 'update_date|desc')"
    )
    mandatoryFilters: Optional[Dict[str, str]] = Field(
        None,
        description="Mandatory filters to restrict access to certain fields"
        " or data"
    )
    userIdFilter: Optional[bool] = Field(
        None,
        description="Whether the current user ID filter should be added to"
        " the listing page"
    )
    fieldElements: Optional[List[FieldElement]] = Field(
        None,
        description="List of field elements to be used in the editor"
    )
    childComponents: Optional[List[str]] = Field(
        None,
        description="List of child components related to the main table"
    )
    type: Literal['master_listing', 'child_listing'] = Field(
        'master_listing',
        description="Editor type: 'master_listing' or 'child_listing'"
        " (default: master_listing)"
    )
    subType: Optional[Literal['array', 'table']] = Field(
        None,
        description="Relationship storage type for child listings"
    )
    array_name: Optional[str] = Field(
        None,
        description="Attribute name for the 'array' type child listing"
    )
    parentKeyNames: Optional[List[ParentKeyName]] = Field(
        None,
        description="Parent Key Names to establish relationship between"
        " parent and child tables"
    )
    parentUrl: Optional[str] = Field(
        None,
        description="Parent URL (seen in some child listing configurations)"
    )
    allow_duplicates: Optional[bool] = Field(
        None,
        description="Allow duplicates in the child listing items creation"
    )
    # Hooks / Functions
    dbListPreRead: Optional[List[str]] = Field(
        None,
        description="Specific functions to execute before reading list data")
    dbListPostRead: Optional[List[str]] = Field(
        None,
        description="Specific functions to execute after reading list data")
    dbPreRead: Optional[List[str]] = Field(
        None,
        description="Specific functions to execute before reading form data")
    dbPostRead: Optional[List[str]] = Field(
        None,
        description="Specific functions to execute after reading form data")
    dbPreValidations: Optional[List[str]] = Field(
        None,
        description="Specific functions to execute before delete validation")
    validations: Optional[List[str]] = Field(
        None,
        description="Specific functions to execute for validation before"
        " write")
    dbPreWrite: Optional[List[str]] = Field(
        None, description="Specific functions to execute before writing"
        " (after validation)")
    dbPostWrite: Optional[List[str]] = Field(
        None,
        description="Specific functions to execute after successful write")


class BackendCrudEditorConfig(BaseModel):
    """
    Configuration for the Generic CRUD Editor (Backend)
    These settings are used exclusively by the backend to handle database
    operations.
    """
    table_name: str = Field(
        ...,
        description="The physical table name in the database")
    creation_pk_name: Optional[str] = Field(
        None, description="Column/attribute name used as the primary"
        " key during creation"
    )
    projection_exclusion: Optional[List[str]] = Field(
        None,
        description="List of attributes to exclude from the API response"
        " (projection)"
    )
    email_verification: Optional[List[str]] = Field(
        None,
        description="List of attributes to be validated as email addresses"
    )
    passwords: Optional[List[str]] = Field(
        None,
        description="List of attributes to be treated/validated as passwords"
    )
    mandatory_fields: Optional[List[str]] = Field(
        None,
        description="List of attributes that must be present (required)"
    )
    additional_query_params: Optional[List[str]] = Field(
        None, description="List of attributes to be used in query"
        " operations other than the primary key"
    )
    specific_function: Optional[str] = Field(
        None, description="Name of a specific function to execute before/after"
        " DB operations"
    )
    notes: Optional[str] = Field(
        None, description="Optional notes or comments about the configuration"
        " (seen in examples)"
    )


# Important for checking recursive references
# (if using Pydantic v1, use update_forward_refs())
# If using Pydantic v2+, this is often handled automatically if using future
# annotations, but calling model_rebuild() is safer if strict.

try:
    FieldElement.model_rebuild()
    FrontendCrudEditorConfig.model_rebuild()
except AttributeError:
    # Pydantic v1 fallback
    FieldElement.update_forward_refs()
    FrontendCrudEditorConfig.update_forward_refs()
