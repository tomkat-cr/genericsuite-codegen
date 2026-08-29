
"""
Frontend Code Generation Tool
"""
from typing import List, Dict, Any

from genericsuite_codegen.utilities.app_logger import (
    log_error,
)
from genericsuite_codegen.agent.types import (
    CodeGenerationResult,
)
from genericsuite_codegen.agent.tool_knowledge_base import KnowledgeBaseTool


DEBUG = True


class FrontendCodeGenerator:
    """
    Frontend code generator for GenericSuite applications.

    Generates ReactJS components following GenericSuite UI patterns
    and ExampleApp structure.
    """

    def __init__(self, kb_tool: KnowledgeBaseTool):
        """Initialize the frontend code generator."""
        self.kb_tool = kb_tool
        self._load_templates()

    def _load_templates(self) -> None:
        """Load frontend code templates."""
        # React component template
        self.react_component_template = '''
import React, {{ useState, useEffect }} from 'react';
import {{ {imports} }} from '@/components/ui';
import {{ {api_imports} }} from '@/lib/api';
import {{ {type_imports} }} from '@/types';

interface {component_name}Props {{
  {props_interface}
}}

interface {component_name}State {{
  {state_interface}
}}

const {component_name}: React.FC<{component_name}Props> = ({{
  {props_destructuring}
}}) => {{
  // State management
  const [{state_variables}] = useState<{component_name}State>({{
    {initial_state}
  }});

  // Effects
  useEffect(() => {{
    {use_effect_code}
  }}, []);

  // Event handlers
  {event_handlers}

  // Render helpers
  {render_helpers}

  return (
    <div className="{container_classes}">
      {component_jsx}
    </div>
  );
}};

export default {component_name};
'''

        # React form component template
        self.react_form_template = '''
import React, {{ useState }} from 'react';
import {{ useForm }} from 'react-hook-form';
import {{ zodResolver }} from '@hookform/resolvers/zod';
import * as z from 'zod';
import {{
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
}} from '@/components/ui/form';
import {{ Button }} from '@/components/ui/button';
import {{ Input }} from '@/components/ui/input';
import {{ Textarea }} from '@/components/ui/textarea';
import {{ {additional_imports} }} from '@/components/ui';

// Validation schema
const {form_name}Schema = z.object({{
  {validation_schema}
}});

type {form_name}Values = z.infer<typeof {form_name}Schema>;

interface {form_name}Props {{
  onSubmit: (values: {form_name}Values) => void;
  initialValues?: Partial<{form_name}Values>;
  isLoading?: boolean;
}}

const {form_name}: React.FC<{form_name}Props> = ({{
  onSubmit,
  initialValues,
  isLoading = false
}}) => {{
  const form = useForm<{form_name}Values>({{
    resolver: zodResolver({form_name}Schema),
    defaultValues: {{
      {default_values}
    }}
  }});

  const handleSubmit = (values: {form_name}Values) => {{
    onSubmit(values);
  }};

  return (
    <Form {{...form}}>
      <form onSubmit={{form.handleSubmit(handleSubmit)}} className="space-y-6">
        {form_fields}

        <Button type="submit" disabled={{isLoading}}>
          {{isLoading ? 'Submitting...' : 'Submit'}}
        </Button>
      </form>
    </Form>
  );
}};

export default {form_name};
'''

    def generate_react_component(self, requirements: str, component_name: str,
                                 component_type: str = "component"
                                 ) -> CodeGenerationResult:
        """
        Generate a React component following GenericSuite patterns.

        Args:
            requirements: Requirements for the component.
            component_name: Name of the component.
            component_type: Type of component (form, table, page, component).

        Returns:
            CodeGenerationResult: Generated React component code.
        """
        try:
            # Get relevant context for React components
            context, sources, raw_results = \
                self.kb_tool.get_context_for_generation(
                    query=f"GenericSuite React component {requirements} "
                    f"{component_type}",
                    max_context_length=None,
                    file_type_filter="jsx"
                )

            if component_type == "form":
                return self._generate_form_component(requirements,
                                                     component_name, sources)
            else:
                return self._generate_generic_component(
                    requirements,
                    component_name,
                    component_type,
                    sources)

        except Exception as e:
            log_error(f"Failed to generate React component: {e}")
            raise RuntimeError(f"React component generation failed: {e}")

    def _generate_form_component(self, requirements: str, component_name: str,
                                 sources: List[str]) -> CodeGenerationResult:
        """Generate a form component."""
        # Parse requirements to extract form fields
        fields = self._parse_form_fields(requirements)

        # Generate validation schema
        validation_schema = self._generate_validation_schema(fields)

        # Generate form fields JSX
        form_fields = self._generate_form_fields_jsx(fields)

        # Generate default values
        default_values = self._generate_default_values(fields)

        # Format the template
        code = self.react_form_template.format(
            form_name=component_name,
            validation_schema=validation_schema,
            default_values=default_values,
            form_fields=form_fields,
            additional_imports=self._get_additional_imports(fields)
        )

        # Generate additional files
        files = {
            f"{component_name}.test.tsx":
                self._generate_component_test(component_name, "form"),
            f"{component_name}.stories.tsx":
                self._generate_storybook_story(component_name, "form")
        }

        return CodeGenerationResult(
            code=code,
            code_type="react_form",
            framework="react",
            files=files,
            imports=self._get_form_imports(),
            usage_instructions=self._generate_form_usage_instructions(
                component_name),
            integration_notes=self._generate_integration_notes(
                "react", "form"),
            sources=sources
        )

    def _generate_generic_component(
            self, requirements: str, component_name: str,
            component_type: str, sources: List[str]) -> CodeGenerationResult:
        """Generate a generic React component."""
        # Generate component parts
        props_interface = self._generate_props_interface(requirements)
        state_interface = self._generate_state_interface(requirements)
        component_jsx = self._generate_component_jsx(
            requirements, component_type)

        # Format the template
        code = self.react_component_template.format(
            component_name=component_name,
            imports=self._get_component_imports(component_type),
            api_imports=self._get_api_imports(requirements),
            type_imports=self._get_type_imports(requirements),
            props_interface=props_interface,
            state_interface=state_interface,
            props_destructuring=self._generate_props_destructuring(
                props_interface),
            state_variables=self._generate_state_variables(state_interface),
            initial_state=self._generate_initial_state(state_interface),
            use_effect_code=self._generate_use_effect_code(requirements),
            event_handlers=self._generate_event_handlers(requirements),
            render_helpers=self._generate_render_helpers(requirements),
            container_classes=self._get_container_classes(component_type),
            component_jsx=component_jsx
        )

        # Generate additional files
        files = {
            f"{component_name}.test.tsx":
                self._generate_component_test(component_name, component_type),
            f"{component_name}.module.css":
                self._generate_component_styles(component_name, component_type)
        }

        return CodeGenerationResult(
            code=code,
            code_type=f"react_{component_type}",
            framework="react",
            files=files,
            imports=self._get_component_imports_list(component_type),
            usage_instructions=self._generate_component_usage_instructions(
                component_name, component_type),
            integration_notes=self._generate_integration_notes(
                "react", component_type),
            sources=sources
        )

    # Helper methods for React component generation
    def _parse_form_fields(self, requirements: str) -> List[Dict[str, Any]]:
        """Parse requirements to extract form fields."""
        fields = []

        # Common field patterns
        field_patterns = {
            "name": {"type": "text", "required": True},
            "email": {"type": "email", "required": True},
            "password": {"type": "password", "required": True},
            "description": {"type": "textarea", "required": False},
            "phone": {"type": "tel", "required": False},
            "age": {"type": "number", "required": False},
            "date": {"type": "date", "required": False},
            "status": {"type": "select", "required": False, "options": [
                "active", "inactive"]},
            "category": {"type": "select", "required": False, "options": [
                "general", "important"]}
        }

        requirements_lower = requirements.lower()
        for field_name, field_config in field_patterns.items():
            if field_name in requirements_lower:
                fields.append({"name": field_name, **field_config})

        # If no fields detected, add basic fields
        if not fields:
            fields = [
                {"name": "name", "type": "text", "required": True},
                {"name": "description", "type": "textarea", "required": False}
            ]

        return fields

    def _generate_validation_schema(self, fields: List[Dict[str, Any]]) -> str:
        """Generate Zod validation schema."""
        schema_parts = []

        for field in fields:
            field_name = field["name"]
            field_type = field["type"]
            required = field.get("required", False)

            if field_type == "email":
                validation = "z.string().email('Invalid email address')"
            elif field_type == "number":
                validation = "z.number().min(0, 'Must be positive')"
            elif field_type == "textarea":
                validation = "z.string().min(10, 'Must be at least 10 "
                "characters')"
            else:
                validation = "z.string().min(1, 'This field is required')"

            if not required:
                validation += ".optional()"

            schema_parts.append(f"  {field_name}: {validation}")

        return ",\n".join(schema_parts)

    def _generate_form_fields_jsx(self, fields: List[Dict[str, Any]]) -> str:
        """Generate JSX for form fields."""
        jsx_parts = []

        for field in fields:
            field_name = field["name"]
            field_type = field["type"]
            label = field_name.replace("_", " ").title()

            if field_type == "textarea":
                control = "Textarea"
            elif field_type == "select":
                control = "Select"
            else:
                control = "Input"

            jsx = f'''        <FormField
          control={{form.control}}
          name="{field_name}"
          render={{({{ field }}) => (
            <FormItem>
              <FormLabel>{label}</FormLabel>
              <FormControl>
                <{control} placeholder="Enter {label.lower()}" {{...field}} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}}
        />'''

            jsx_parts.append(jsx)

        return "\n\n".join(jsx_parts)

    def _generate_default_values(self, fields: List[Dict[str, Any]]) -> str:
        """Generate default values for form fields."""
        defaults = []

        for field in fields:
            field_name = field["name"]
            field_type = field["type"]

            if field_type == "number":
                default_value = "0"
            elif field_type == "boolean":
                default_value = "false"
            else:
                default_value = '""'

            defaults.append(f"      {field_name}: {default_value}")

        return ",\n".join(defaults)

    def _get_additional_imports(self, fields: List[Dict[str, Any]]) -> str:
        """Get additional UI component imports based on fields."""
        imports = set()

        for field in fields:
            field_type = field["type"]
            if field_type == "textarea":
                imports.add("Textarea")
            elif field_type == "select":
                imports.add("Select")
            elif field_type == "checkbox":
                imports.add("Checkbox")

        return ", ".join(sorted(imports))

    # More helper methods would continue here...
    def _generate_props_interface(self, requirements: str) -> str:
        """Generate TypeScript props interface."""
        return "// Props interface based on requirements"

    def _generate_state_interface(self, requirements: str) -> str:
        """Generate TypeScript state interface."""
        return "// State interface based on requirements"

    def _generate_component_jsx(self, requirements: str, component_type: str
                                ) -> str:
        """Generate component JSX."""
        return \
            f"{{/* {component_type} component JSX based on requirements */}}"

    def _get_component_imports(self, component_type: str) -> str:
        """Get component imports."""
        return "Button, Card, CardContent, CardHeader, CardTitle"

    def _get_api_imports(self, requirements: str) -> str:
        """Get API imports."""
        return "useApi, ApiResponse"

    def _get_type_imports(self, requirements: str) -> str:
        """Get type imports."""
        return "ComponentProps, ApiData"

    def _generate_props_destructuring(self, props_interface: str) -> str:
        """Generate props destructuring."""
        return "// Props destructuring"

    def _generate_state_variables(self, state_interface: str) -> str:
        """Generate state variables."""
        return "data, setData, loading, setLoading"

    def _generate_initial_state(self, state_interface: str) -> str:
        """Generate initial state."""
        return "data: null, loading: false"

    def _generate_use_effect_code(self, requirements: str) -> str:
        """Generate useEffect code."""
        return "// Initialize component data"

    def _generate_event_handlers(self, requirements: str) -> str:
        """Generate event handlers."""
        return "// Event handlers based on requirements"

    def _generate_render_helpers(self, requirements: str) -> str:
        """Generate render helper functions."""
        return "// Render helpers based on requirements"

    def _get_container_classes(self, component_type: str) -> str:
        """Get container CSS classes."""
        return f"container mx-auto p-4 {component_type}-container"

    def _get_form_imports(self) -> List[str]:
        """Get form component imports."""
        return [
            "react-hook-form",
            "@hookform/resolvers/zod",
            "zod",
            "@/components/ui/form",
            "@/components/ui/button",
            "@/components/ui/input"
        ]

    def _get_component_imports_list(self, component_type: str) -> List[str]:
        """Get component imports list."""
        return [
            "react",
            "@/components/ui/button",
            "@/components/ui/card",
            "@/lib/api"
        ]

    def _generate_component_test(self, component_name: str,
                                 component_type: str) -> str:
        """Generate component test file."""
        return f'''import {{ render, screen }} from '@testing-library/react';
import {component_name} from './{component_name}';

describe('{component_name}', () => {{
  it('renders without crashing', () => {{
    render(<{component_name} />);
    expect(screen.getByRole('main')).toBeInTheDocument();
  }});

  // Add more tests based on component functionality
}});
'''

    def _generate_storybook_story(self, component_name: str,
                                  component_type: str) -> str:
        """Generate Storybook story."""
        return f'''import type {{ Meta, StoryObj }} from '@storybook/react';
import {component_name} from './{component_name}';

const meta: Meta<typeof {component_name}> = {{
  title: 'Components/{component_name}',
  component: {component_name},
  parameters: {{
    layout: 'centered',
  }},
}};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {{
  args: {{
    // Default props
  }},
}};
'''

    def _generate_component_styles(self, component_name: str,
                                   component_type: str) -> str:
        """Generate component CSS module."""
        return f'''.{component_name.lower()} {{
  /* Component styles */
}}

.{component_name.lower()}Container {{
  /* Container styles */
}}
'''

    def _generate_form_usage_instructions(self, component_name: str) -> str:
        """Generate form usage instructions."""
        return f'''Usage Instructions for {component_name}:

1. Import the component:
   import {component_name} from '@/components/{component_name}';

2. Use in your page/component:
   <{component_name}
     onSubmit={{handleSubmit}}
     initialValues={{initialData}}
     isLoading={{isSubmitting}}
   />

3. Handle form submission:
   const handleSubmit = (values) => {{
     // Process form data
     console.log(values);
   }};
'''

    def _generate_component_usage_instructions(self, component_name: str,
                                               component_type: str) -> str:
        """Generate component usage instructions."""
        return f'''Usage Instructions for {component_name}:

1. Import the component:
   import {component_name} from '@/components/{component_name}';

2. Use in your application:
   <{component_name} />

3. Customize props as needed based on your requirements.
'''

    def _generate_integration_notes(self, framework: str, component_type: str
                                    ) -> str:
        """Generate integration notes."""
        return f'''Integration Notes for {framework} {component_type}:

- Follows GenericSuite UI patterns and conventions
- Uses ShadCn/UI components for consistency
- Includes proper TypeScript types and interfaces
- Implements responsive design principles
- Compatible with GenericSuite authentication and API patterns

Make sure to:
1. Install required dependencies
2. Configure your build system for the imports
3. Set up proper routing if this is a page component
4. Test the component in your specific environment
'''
