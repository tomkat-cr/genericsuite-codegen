// CrudEditorConfigInterface.ts
// 2025-12-26 | CR

// This file defines the TypeScript interfaces to validate the frontend and backend JSON
// configurations for the Generic CRUD Editor.

/**
 * Supported field types
 */
export type FieldType =
    | 'text'
    | 'textarea'
    | 'number'
    | 'integer'
    | 'date'
    | 'datetime-local'
    | 'array'
    | 'email'
    | 'label'
    | 'h1' | 'h2' | 'h3' | 'h4' | 'h5' | 'h6'
    | 'hr'
    | '_id'
    | 'select'
    | 'select_table'
    | 'select_component'
    | 'suggestion_dropdown'
    | 'component'

/**
 * Definition of a parent key relationship
 */
export interface ParentKeyName {
    /** The name of the parameter to be used in the backend API call to identify the parent table Primary Key */
    parameterName: string
    /** The name of the parent table's primary key column/attribute */
    parentElementName: string
}

/**
 * individual field configuration for the CRUD editor
 */
export interface FieldElement {
    /** The name of the field element */
    name: string
    /** The label displayed for the field */
    label: string
    /** The data type of the field */
    type: FieldType | string
    /** Whether the field is required for submission. Default: false */
    required?: boolean
    /** Whether the field should be displayed in the listing page. Default: false */
    listing?: boolean
    /** Whether the field should be read-only. Default: false */
    readonly?: boolean
    /** The default value to use for the field */
    default_value?: string | number
    /** Whether the field should be hidden from display. Default: false */
    hidden?: boolean
    /** Whether the field is the primary key of the table. Default: false unless the field is of type '_id' */
    primaryKey?: boolean
    /** Formula function name to calculate value from other fields */
    formula?: string
    /** Select specific: ID for predefined select options (e.g., "TRUE_FALSE") */
    select_elements?: string
    // Suggestion Dropdown specific attributes
    /** Field name in API response for the selected item Key */
    suggestion_id_fieldname?: string
    /** Field name in API response for the selected item Name */
    suggestion_desc_fieldname?: string
    /** API URL for suggestions */
    filter_api_url?: string
    /** HTTP method for suggestion API */
    filter_api_request_method?: string
    /** Parameter name for the search term */
    filter_search_param_name?: string
    /** Additional parameters for the suggestion API */
    filter_search_other_param?: Record<string, string>
    /** Fields to populate from the selected item */
    autocomplete_fields?: Record<string, string>
    // Special Button attributes
    /** Enable ChatBot button */
    chatbot_popup?: boolean
    /** Text prompt for ChatBot button (%s replaced by field value) */
    chatbot_prompt?: string
    /** Enable Google Search button */
    google_popup?: boolean
    /** Text prompt for Google Search button (%s replaced by field value) */
    google_prompt?: string
    /** Component name for auxiliary button (e.g. ChatBotButton) */
    aux_component?: string
    /** Enable UUID generation for this field */
    uuid_generator?: boolean
}

/**
 * Configuration for the Generic CRUD Editor (Frontend)
 */
export interface FrontendCrudEditorConfig {
    /** The ReactJS router URL for the CRUD editor */
    baseUrl: string
    /** The title of the CRUD editor, used in the listing page */
    title: string
    /** The name of the CRUD editor component, used as a title in the form submission page */
    name: string
    /** The ReactJS component name used by the CRUD editor */
    component: string
    /** The API endpoint URL that the CRUD editor interacts with */
    dbApiUrl: string
    /** Primary Key parameter name for API calls (default: id) */
    primaryKeyName?: string
    /** The default order for sorting data in the listing page (e.g., "update_date|desc") */
    defaultOrder?: string
    /** Mandatory filters to restrict access to certain fields or data */
    mandatoryFilters?: Record<string, string>
    /** Whether the current user ID filter should be added to the listing page */
    userIdFilter?: boolean
    /** List of field elements to be used in the editor */
    fieldElements: FieldElement[]
    /** List of child components related to the main table */
    childComponents?: string[]
    /** Editor type: 'master_listing' or 'child_listing' (default: master_listing) */
    type?: 'master_listing' | 'child_listing'
    /** Relationship storage type for child listings */
    subType?: 'array' | 'table'
    /** Attribute name for the 'array' type child listing */
    array_name?: string
    /** Parent Key Names to establish relationship between parent and child tables */
    parentKeyNames?: ParentKeyName[]
    /** The parent table URL, to be used in certain frontend specific functions */
    parentUrl?: string
    /** Allow duplicates in the child listing items creation */
    allow_duplicates?: boolean
    /** Specific functions to execute before reading list data */
    dbListPreRead?: string[]
    /** Specific functions to execute after reading list data */
    dbListPostRead?: string[]
    /** Specific functions to execute before reading form data */
    dbPreRead?: string[]
    /** Specific functions to execute after reading form data */
    dbPostRead?: string[]
    /** Specific functions to execute before delete validation */
    dbPreValidations?: string[]
    /** Specific functions to execute for validation before write */
    validations?: string[]
    /** Specific functions to execute before writing (after validation) */
    dbPreWrite?: string[]
    /** Specific functions to execute after successful write */
    dbPostWrite?: string[]
}

/**
 * Configuration for the Generic CRUD Editor (Backend)
 * These settings are used exclusively by the backend to handle database operations.
 */
export interface BackendCrudEditorConfig {
    /** The physical table name in the database */
    table_name: string
    /** Column/attribute name used as the primary key during creation. Default: _id */
    creation_pk_name?: string
    /** List of attributes to exclude from the API response (projection) */
    projection_exclusion?: string[]
    /** List of attributes to be validated as email addresses */
    email_verification?: string[]
    /** List of attributes to be treated/validated as passwords */
    passwords?: string[]
    /** List of attributes that must be present (required) */
    mandatory_fields?: string[]
    /** List of attributes to be used in query operations other than the primary key */
    additional_query_params?: string[]
    /** Name of a specific function to execute before/after DB operations */
    specific_function?: string

    /** Optional notes or comments about the configuration (seen in examples) */
    notes?: string
}
