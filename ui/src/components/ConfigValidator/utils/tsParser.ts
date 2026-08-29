
export interface SchemaField {
    name: string;
    type: string;
    required: boolean;
    isArray: boolean;
    description?: string;
    allowedValues?: string[]; // For unions of string literals
}

export interface SchemaInterface {
    name: string;
    fields: SchemaField[];
    description?: string;
}

export interface SchemaType {
    name: string;
    type: string; // Alias to another type or union
    options?: string[]; // If it's a union of strings
    description?: string;
}

export interface ParsedSchema {
    interfaces: Record<string, SchemaInterface>;
    types: Record<string, SchemaType>;
    referencedInterfaces: Array<string>;
}

export const parseTsSchema = (tsContent: string): ParsedSchema => {
    const lines = tsContent.split('\n');
    const schema: ParsedSchema = { interfaces: {}, types: {}, referencedInterfaces: [] };

    let currentInterface: SchemaInterface | null = null;
    let currentDescription: string[] = [];

    // Helper to extract description from JSDoc
    const getCleanDescription = () => {
        if (currentDescription.length === 0) return undefined;
        const desc = currentDescription
            .map(line => line.replace(/^\s*\*\s?/, '').trim())
            .filter(line => line !== '/' && line !== '')
            .join(' ');
        currentDescription = [];
        return desc || undefined;
    };

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();

        // Capture JSDoc comments
        if (line.startsWith('/**') || line.startsWith('*')) {
            if (line.startsWith('/**')) currentDescription = [];
            if (!line.endsWith('*/')) {
                currentDescription.push(line);
            } else {
                currentDescription.push(line.replace('*/', ''));
            }
            continue;
        }

        // Interface Start
        const interfaceMatch = line.match(/export\s+interface\s+(\w+)\s*{/);
        if (interfaceMatch) {
            currentInterface = {
                name: interfaceMatch[1],
                fields: [],
                description: getCleanDescription()
            };
            continue;
        }

        // Interface End
        if (line === '}' && currentInterface) {
            schema.interfaces[currentInterface.name] = currentInterface;
            currentInterface = null;
            continue;
        }

        // Type Definition (Enum-like string union)
        // export type FieldType = 'text' | 'textarea' ...
        // We need to handle multi-line types
        const typeMatch = line.match(/export\s+type\s+(\w+)\s*=\s*(.*)/);
        if (typeMatch) {
            const typeName = typeMatch[1];
            let typeBody = typeMatch[2];

            // Multiline collection
            let j = i;
            while (
                !typeBody.includes(';')
                && !lines[j].includes(';')
                && lines[j].trim() !== ''
                && j < lines.length - 1
            ) {
                j++;
                // Clean up comments if mixed in, but usually just pipes
                typeBody += ' ' + lines[j].trim();
            }
            i = j; // Advance main loop

            // Clean up type body
            typeBody = typeBody.replace(';', '').trim();

            // Check if union of strings
            if (typeBody.includes('|')) {
                const options = typeBody.split('|').map(s => s.trim().replace(/^['"]|['"]$/g, ''));
                schema.types[typeName] = {
                    name: typeName,
                    type: 'union',
                    options: options.filter(o => o),
                    description: getCleanDescription()
                };
            } else {
                schema.types[typeName] = {
                    name: typeName,
                    type: typeBody,
                    description: getCleanDescription()
                };
            }
            continue;
        }

        // Field Definition inside Interface
        if (currentInterface) {
            // name?: string
            // name: string
            // name: FieldType | string
            const fieldMatch = line.match(/^(\w+)(\?)?:\s*(.+)/);
            if (fieldMatch) {
                const name = fieldMatch[1];
                const isOptional = !!fieldMatch[2];
                let typeStr = fieldMatch[3].trim();

                // Handle Inline Comments? usually not in provided file structure, but good to be safe.
                // The provided file has comments above, not inline usually.

                // Determine array
                const isArray = typeStr.endsWith('[]');
                if (isArray) {
                    typeStr = typeStr.slice(0, -2);
                }

                // Clean trailing types like ' | string', '; at end'
                // The sample file has things like 'FieldType | string' or 'Record<string, string>'

                // Simple stripping of ;
                if (typeStr.endsWith(';')) typeStr = typeStr.slice(0, -1);

                // Handle complex types slightly?
                // For now, store raw type string, validator will interpret.

                currentInterface.fields.push({
                    name,
                    required: !isOptional,
                    type: typeStr,
                    isArray,
                    description: getCleanDescription()
                });

                // Check if the type is an already added interface name
                if (schema.interfaces[typeStr]) {
                    schema.referencedInterfaces.push(typeStr);
                }
            }
        }
    }

    return schema;
};
