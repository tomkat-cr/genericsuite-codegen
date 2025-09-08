# shadcn/ui Setup Instructions

This project uses shadcn/ui for UI components. Follow these steps to set up the components properly.

## Initial Setup

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Initialize shadcn/ui** (if not already done):
   ```bash
   npm run shadcn:init
   # or directly: npx shadcn@latest init
   ```

   This will:
   - Install core dependencies (`class-variance-authority`, `clsx`, `tailwind-merge`, `lucide-react`, `tailwindcss-animate`)
   - Create `components.json` configuration file
   - Set up the `cn` utility function
   - Configure CSS variables for theming

## Adding Components

Use the shadcn CLI to add components as needed:

```bash
# Add individual components
npm run shadcn:add button
npm run shadcn:add card
npm run shadcn:add form
npm run shadcn:add input
npm run shadcn:add dialog

# Or directly
npx shadcn@latest add button card form input dialog
```

## Available Components

Common components you might need:

- **Layout**: `card`, `separator`, `sheet`, `dialog`
- **Forms**: `form`, `input`, `textarea`, `button`, `checkbox`, `radio-group`, `select`
- **Navigation**: `navigation-menu`, `breadcrumb`, `dropdown-menu`, `context-menu`
- **Feedback**: `alert`, `alert-dialog`, `toast`, `sonner`, `progress`
- **Data Display**: `table`, `badge`, `avatar`, `skeleton`
- **Interactive**: `accordion`, `collapsible`, `tabs`, `toggle`, `switch`

## Manual Installation (if needed)

If you need to install a component manually:

1. Install the required Radix UI dependency:
   ```bash
   npm install @radix-ui/react-[component-name]
   ```

2. Copy the component code from [shadcn/ui components](https://ui.shadcn.com/docs/components)

3. Place it in `src/components/ui/[component-name].tsx`

## Project Structure

After setup, your project should have:

```
src/
├── components/
│   └── ui/           # shadcn/ui components
├── lib/
│   └── utils.ts      # cn utility function
└── styles/
    └── globals.css   # Global styles with CSS variables
```

## Configuration Files

- `components.json` - shadcn/ui configuration
- `tailwind.config.js` - Tailwind CSS configuration with shadcn/ui presets
- `src/lib/utils.ts` - Utility functions including the `cn` helper

## Notes

- **No need for individual @radix-ui packages** in package.json - shadcn CLI handles this
- **Components are copied to your project** - you own the code and can customize it
- **Use the CLI whenever possible** - it ensures proper setup and dependencies
- **CSS variables are used for theming** - customize in `globals.css`

## Troubleshooting

If you encounter issues:

1. Make sure you've run `npx shadcn@latest init`
2. Check that `components.json` exists and is properly configured
3. Verify that Tailwind CSS is set up correctly
4. Ensure the `cn` utility function exists in `src/lib/utils.ts`

For more information, visit the [official shadcn/ui documentation](https://ui.shadcn.com/).