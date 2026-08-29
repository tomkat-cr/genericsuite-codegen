
import React, { useEffect, useState } from 'react';
import { validateConfig, type ValidationError } from './utils/jsonValidator';
import { parseTsSchema, type ParsedSchema } from './utils/tsParser';

const schemaFilePath = '/CrudEditorConfigInterface.ts';
console.log('>> schemaFilePath:', schemaFilePath);

function ConfigValidator() {
  const [schema, setSchema] = useState<ParsedSchema | null>(null);
  const [jsonInput, setJsonInput] = useState('');
  const [validationErrors, setValidationErrors] = useState<ValidationError[]>([]);
  const [isValid, setIsValid] = useState<boolean | null>(null);
  const [configType, setConfigType] = useState('FrontendCrudEditorConfig');
  const [loading, setLoading] = useState(true);
  const [parseError, setParseError] = useState('');

  useEffect(() => {
    // Fetch the TS definition file
    fetch(schemaFilePath)
      .then(res => {
        if (!res.ok) throw new Error("Failed to load schema file");
        return res.text();
      })
      .then(text => {
        try {
          const parsed = parseTsSchema(text);
          setSchema(parsed);
          setLoading(false);
        } catch (e) {
             console.error(e);
             setParseError('Failed to parse TypeScript schema definition.');
             setLoading(false);
        }
      })
      .catch(err => {
        console.error(err);
        setParseError(`Error loading schema: ${err.message}`);
        setLoading(false);
      });
  }, []);

  const handleValidation = () => {
    if (!schema) return;
    setIsValid(null);
    setValidationErrors([]);

    let parsedJson;
    try {
      parsedJson = JSON.parse(jsonInput);
    } catch (e) {
      setIsValid(false);
      setValidationErrors([{
        path: 'JSON Parser',
        message: 'Invalid JSON syntax',
        severity: 'error'
      }]);
      return;
    }

    const errors = validateConfig(schema, configType, parsedJson);
    setValidationErrors(errors);
    setIsValid(errors.length === 0);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (evt) => {
      setJsonInput(evt.target?.result as string);
    };
    reader.readAsText(file);
  };

console.log('>> schema:',schema);

  return (
    <div className="min-h-screen p-8 font-sans">
      <div className="max-w-4xl mx-auto bg-white rounded-lg border border-gray-200 shadow-md p-6">
        <h1 className="text-2xl font-bold mb-4 text-gray-800">
          CRUD Editor JSON Config Files Validator
        </h1>
        
        {loading && <p className="text-gray-600">Loading schema definition...</p>}
        {parseError && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            <p>{parseError}</p>
          </div>
        )}

        {!loading && !parseError && (
          <>
            <div className="mb-6">
              <label className="block text-gray-700 text-sm font-bold mb-2">
                Configuration Type
              </label>
              <select 
                value={configType}
                onChange={(e) => setConfigType(e.target.value)}
                className="shadow border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
              >
                {schema?.interfaces && Object.keys(schema.interfaces)
                  .filter(name => !schema.referencedInterfaces.includes(name))
                  .map(name => (
                  <option key={name} value={name}>{name}</option>
                ))}
              </select>
            </div>

            <div className="mb-6">
              <label className="block text-gray-700 text-sm font-bold mb-2">
                JSON Configuration
              </label>
              <div className="flex gap-4 mb-2">
                 <input 
                   type="file" 
                   accept=".json"
                   onChange={handleFileUpload}
                   className="block w-full text-sm text-gray-500
                     file:mr-4 file:py-2 file:px-4
                     file:rounded-full file:border-0
                     file:text-sm file:font-semibold
                     file:bg-blue-50 file:text-blue-700
                     hover:file:bg-blue-100"
                 />
              </div>
              <textarea
                value={jsonInput}
                onChange={(e) => setJsonInput(e.target.value)}
                className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline font-mono text-sm"
                rows={15}
                placeholder="Paste your JSON configuration here..."
              />
            </div>

            <button
              onClick={handleValidation}
              className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
            >
              Validate
            </button>

            {/* Results */}
            <div className="mt-8">
              {isValid === true && (
                <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded">
                  <strong className="font-bold">Success! </strong>
                  <span className="block sm:inline">The configuration is valid against {configType}.</span>
                </div>
              )}

              {isValid === false && (
                <div className="bg-red-50 border border-red-200 rounded p-4">
                  <h3 className="text-lg font-bold text-red-800 mb-2">Validation Errors ({validationErrors.length})</h3>
                  <ul className="list-disc pl-5">
                    {validationErrors.map((err, idx) => (
                      <li key={idx} className="text-red-700 mb-1">
                        <span className="font-mono bg-red-100 px-1 rounded mr-2">{err.path || 'Root'}</span>
                        {err.message}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default ConfigValidator;
