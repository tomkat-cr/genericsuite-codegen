import type { ParsedSchema, SchemaField } from './tsParser';

export interface ValidationError {
    path: string;
    message: string;
    severity: 'error' | 'warning';
}

export const validateConfig = (
    schema: ParsedSchema,
    rootInterfaceName: string,
    data: any
): ValidationError[] => {
    const errors: ValidationError[] = [];

    const validateRecursive = (
        interfaceName: string,
        currentData: any,
        path: string
    ) => {
        // Base case: undefined/null checks handled by container usually, but check here
        if (currentData === null || currentData === undefined) {
            return;
        }

        const interfaceDef = schema.interfaces[interfaceName];
        if (!interfaceDef) {
            // Might be a primitive or unknown type passed as interface name
            return;
        }

        // Check required fields and type match
        for (const field of interfaceDef.fields) {
            const fieldPath = path ? `${path}.${field.name}` : field.name;
            const value = currentData[field.name];

            // Required check
            if (field.required && (value === undefined || value === null || value === '')) {
                errors.push({
                    path: fieldPath,
                    message: `Missing required field: ${field.name}`,
                    severity: 'error'
                });
                continue;
            }

            if (value !== undefined && value !== null) {
                validateFieldType(field, value, fieldPath);
            }
        }
    };

    const validateFieldType = (field: SchemaField, value: any, path: string) => {
        if (field.isArray) {
            if (!Array.isArray(value)) {
                errors.push({
                    path: path,
                    message: `Expected array for field ${field.name}`,
                    severity: 'error'
                });
                return;
            }
            // Validate each item
            value.forEach((item, index) => {
                validateSingleType(field.type, item, `${path}[${index}]`);
            });
        } else {
            validateSingleType(field.type, value, path);
        }
    };

    const validateSingleType = (type: string, value: any, path: string) => {
        // Handle Record<string, string>
        if (type.startsWith('Record<')) {
            if (typeof value !== 'object' || Array.isArray(value)) {
                errors.push({ path, message: `Expected object (Record), got ${typeof value}`, severity: 'error' });
                return;
            }
            // For now assume Record<string, string> - could parse deeper if needed
            return;
        }

        // Handle Unions in type string (e.g. "FieldType | string")
        if (type.includes('|')) {
            // Simple check: matches one of the types
            // This is loose, but for "string | number" it works.
            // For "FieldType | string", if it's a string it passes.
            // If it's an enum (FieldType), it also passes if it's a string in TS usually (unless strictly typed).
            // But we should check if it matches specific values if possible.
            return;
        }

        // Check against known Types (Enums/Unions)
        if (schema.types[type]) {
            const typeDef = schema.types[type];
            if (typeDef.type === 'union' && typeDef.options) {
                if (!typeDef.options.includes(value)) {
                    errors.push({
                        path,
                        message: `Invalid value '${value}'. Expected one of: ${typeDef.options.join(', ')}`,
                        severity: 'error'
                    });
                }
            }
            return;
        }

        // Check against known Interfaces (Recursive)
        if (schema.interfaces[type]) {
            if (typeof value !== 'object' || Array.isArray(value)) {
                errors.push({ path, message: `Expected object for type ${type}`, severity: 'error' });
                return;
            }
            validateRecursive(type, value, path);
            return;
        }

        // Primitives
        if (type === 'string') {
            if (typeof value !== 'string') {
                errors.push({ path, message: `Expected string, got ${typeof value}`, severity: 'error' });
            }
        } else if (type === 'number' || type === 'integer') {
            if (typeof value !== 'number') {
                errors.push({ path, message: `Expected number, got ${typeof value}`, severity: 'error' });
            }
        } else if (type === 'boolean') {
            if (typeof value !== 'boolean') {
                errors.push({ path, message: `Expected boolean, got ${typeof value}`, severity: 'error' });
            }
        }
    };

    validateRecursive(rootInterfaceName, data, '');
    return errors;
};
