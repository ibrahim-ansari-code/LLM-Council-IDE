import React, { useState, useRef, useEffect } from 'react'
import Editor from '@monaco-editor/react'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'
import { Send, Code, MessageSquare, Loader2, Folder, File, Play, X, Plus, Trash2 } from 'lucide-react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

function App() {
  const [files, setFiles] = useState([])
  const [openTabs, setOpenTabs] = useState([])
  const [activeTab, setActiveTab] = useState(null)
  const [fileContents, setFileContents] = useState({})
  const [currentPath, setCurrentPath] = useState('')
  
  const [language, setLanguage] = useState('javascript')
  const editorRef = useRef(null)
  
  const [chatMessages, setChatMessages] = useState([])
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [showChat, setShowChat] = useState(true)
  const [currentResponse, setCurrentResponse] = useState(null)
  const [councilTab, setCouncilTab] = useState('final')
  const [conversationId, setConversationId] = useState(null)
  const chatEndRef = useRef(null)
  
  const [showTerminal, setShowTerminal] = useState(false)
  const [terminalOutput, setTerminalOutput] = useState([])
  const [terminalInput, setTerminalInput] = useState('')
  const [isExecuting, setIsExecuting] = useState(false)
  const [programInput, setProgramInput] = useState('')
  const [needsInput, setNeedsInput] = useState(false)

  useEffect(() => {
    loadFiles()
  }, [currentPath])

  useEffect(() => {
    if (activeTab && fileContents[activeTab]) {
      const saveTimeout = setTimeout(() => {
        saveFile(activeTab, fileContents[activeTab])
      }, 1000)
      return () => clearTimeout(saveTimeout)
    }
  }, [fileContents, activeTab])

  const loadFiles = async () => {
    try {
      const response = await axios.get(`${API_BASE}/api/files`, {
        params: { path: currentPath }
      })
      setFiles(response.data.files)
    } catch (error) {
      console.error('Error loading files:', error)
    }
  }

  const openFile = async (filePath) => {
    if (openTabs.includes(filePath)) {
      setActiveTab(filePath)
      return
    }

    try {
      const response = await axios.get(`${API_BASE}/api/files/${filePath}`)
      const newTabs = [...openTabs, filePath]
      setOpenTabs(newTabs)
      setActiveTab(filePath)
      setFileContents(prev => ({
        ...prev,
        [filePath]: response.data.content
      }))
      
      const ext = filePath.split('.').pop()?.toLowerCase()
      const langMap = {
        js: 'javascript', jsx: 'javascript', ts: 'typescript', tsx: 'typescript',
        py: 'python', java: 'java', cpp: 'cpp', c: 'c', go: 'go', rs: 'rust',
        rb: 'ruby', php: 'php', html: 'html', css: 'css', json: 'json',
        yml: 'yaml', yaml: 'yaml', md: 'markdown'
      }
      setLanguage(langMap[ext] || 'plaintext')
    } catch (error) {
      console.error('Error opening file:', error)
    }
  }

  const closeTab = (filePath) => {
    const newTabs = openTabs.filter(t => t !== filePath)
    setOpenTabs(newTabs)
    if (activeTab === filePath) {
      setActiveTab(newTabs.length > 0 ? newTabs[newTabs.length - 1] : null)
    }
    const newContents = { ...fileContents }
    delete newContents[filePath]
    setFileContents(newContents)
  }

  const saveFile = async (filePath, content) => {
    try {
      await axios.post(`${API_BASE}/api/files/${filePath}`, { content })
    } catch (error) {
      console.error('Error saving file:', error)
    }
  }

  const createFile = async (fileName) => {
    const filePath = currentPath ? `${currentPath}/${fileName}` : fileName
    try {
      await axios.post(`${API_BASE}/api/files/${filePath}`, { content: '' })
      await loadFiles()
      await openFile(filePath)
    } catch (error) {
      console.error('Error creating file:', error)
    }
  }

  const deleteFile = async (filePath) => {
    if (!confirm(`Delete ${filePath}?`)) return
    try {
      await axios.delete(`${API_BASE}/api/files/${filePath}`)
      closeTab(filePath)
      await loadFiles()
    } catch (error) {
      console.error('Error deleting file:', error)
    }
  }

  const handleEditorDidMount = (editor) => {
    editorRef.current = editor
  }

  const handleEditorChange = (value) => {
    if (activeTab) {
      setFileContents(prev => ({
        ...prev,
        [activeTab]: value || ''
      }))
    }
  }

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return

    const userMessage = {
      role: 'user',
      content: inputMessage,
      timestamp: new Date().toISOString(),
    }

    setChatMessages((prev) => [...prev, userMessage])
    const message = inputMessage
    setInputMessage('')
    setIsLoading(true)
    setCurrentResponse(null)

    try {
      let writeToFile = null
      
      const patterns = [
        /(?:create|make|write|save)\s+(?:a\s+)?(?:file\s+)?(?:called|named|to)?\s*['"]?([a-zA-Z0-9_./-]+\.(?:py|js|ts|jsx|tsx|html|css|json|md|txt|java|cpp|c|go|rs|rb|php|sh))['"]?/i,
        /(?:create|make|write|save)\s+(?:a\s+)?(?:file\s+)?(?:called|named|to)?\s*['"]?([a-zA-Z0-9_./-]+)['"]?\s+(?:file|\.py|\.js|\.ts)/i,
        /file\s+(?:called|named)?\s*['"]?([a-zA-Z0-9_./-]+\.(?:py|js|ts|jsx|tsx|html|css|json|md|txt))['"]?/i,
        /([a-zA-Z0-9_./-]+\.(?:py|js|ts|jsx|tsx|html|css|json|md|txt))/i,
      ]
      
      for (const pattern of patterns) {
        const match = message.match(pattern)
        if (match && match[1]) {
          writeToFile = match[1].trim()
          writeToFile = writeToFile.replace(/^['"]+|['"]+$/g, '')
          break
        }
      }
      
      if (!writeToFile && (message.toLowerCase().includes('create') || message.toLowerCase().includes('make') || message.toLowerCase().includes('write'))) {
        const words = message.split(/\s+/)
        for (let i = 0; i < words.length; i++) {
          if ((words[i].toLowerCase() === 'create' || words[i].toLowerCase() === 'make' || words[i].toLowerCase() === 'write') && i + 1 < words.length) {
            const nextWord = words[i + 1].replace(/[.,;:!?]/g, '')
            if (nextWord && nextWord.length > 2) {
              writeToFile = nextWord + '.py'
              break
            }
          }
        }
      }


      let context = ''
      const files_in_context = openTabs.length > 0 ? openTabs : []
      if (openTabs.length > 0) {
        const fileContexts = openTabs.map(tab => {
          const content = fileContents[tab] || ''
          return `File: ${tab}\n${content}`
        }).join('\n\n---\n\n')
        context = `Open files:\n\n${fileContexts}`
      }

      let enhancedMessage = message
      if (writeToFile) {
        enhancedMessage = `${message}\n\nIMPORTANT: If you generate any code, you MUST write it to the file "${writeToFile}". Include the complete code in code blocks.`
      }

      const response = await axios.post(`${API_BASE}/api/chat`, {
        message: enhancedMessage,
        context,
        write_to_file: writeToFile,
        conversation_id: conversationId,
      })
      

      if (response.data.conversation_id) {
        setConversationId(response.data.conversation_id)
      }

      setCurrentResponse(response.data)
      
      const assistantMessage = {
        role: 'assistant',
        content: response.data.final_response,
        councilData: response.data,
        fileWritten: response.data.file_written,
        filesWritten: response.data.files_written,
        timestamp: new Date().toISOString(),
      }

      setChatMessages((prev) => [...prev, assistantMessage])
      
      const filesToOpen = response.data.files_written || (response.data.file_written ? [response.data.file_written] : [])
      if (filesToOpen.length > 0) {
        await loadFiles()
        await new Promise(resolve => setTimeout(resolve, 500))
        for (const filePath of filesToOpen) {
          await openFile(filePath)
          try {
            const fileResponse = await axios.get(`${API_BASE}/api/files/${filePath}`)
            setFileContents(prev => ({
              ...prev,
              [filePath]: fileResponse.data.content
            }))
          } catch (e) {
            console.error(`Error reloading file ${filePath}:`, e)
          }
        }
      } else if (files_in_context && files_in_context.length > 0) {
        await loadFiles()
        for (const filePath of files_in_context) {
          try {
            const fileResponse = await axios.get(`${API_BASE}/api/files/${filePath}`)
            setFileContents(prev => ({
              ...prev,
              [filePath]: fileResponse.data.content
            }))
          } catch (e) {
          }
        }
      }
    } catch (error) {
      console.error('Error sending message:', error)
      const errorMessage = {
        role: 'assistant',
        content: `Error: ${error.response?.data?.detail || error.message}`,
        isError: true,
        timestamp: new Date().toISOString(),
      }
      setChatMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const executeCode = async (providedInput = null) => {
    if (!activeTab || isExecuting) return

    let inputData = providedInput
    
    if (needsInput && !inputData && !programInput) {
      setShowTerminal(true)
      return
    }
    
    if (!inputData && programInput) {
      inputData = programInput
    }

    setIsExecuting(true)
    const filePath = activeTab
    const ext = filePath.split('.').pop()?.toLowerCase()
    const langMap = {
      js: 'javascript', jsx: 'javascript', ts: 'typescript', tsx: 'typescript',
      py: 'python', sh: 'bash'
    }
    const lang = langMap[ext] || 'bash'

    try {
      const requestBody = {
        file_path: filePath,
        language: lang,
      }
      
      if (inputData !== null) {
        requestBody.input_data = inputData
      }

      const response = await axios.post(`${API_BASE}/api/execute`, requestBody)

      const output = {
        type: 'command',
        command: `Running ${filePath}${inputData !== null ? ` (with input)` : ''}`,
        stdout: response.data.stdout,
        stderr: response.data.stderr,
        success: response.data.success,
        timestamp: new Date().toISOString(),
      }

      setTerminalOutput(prev => [...prev, output])
      setShowTerminal(true)
      
      if (inputData !== null) {
        setProgramInput('')
        setNeedsInput(false)
      }
    } catch (error) {
      const errorMsg = error.response?.data?.detail || error.response?.data?.message || error.message || 'Unknown error'
      const output = {
        type: 'error',
        error: typeof errorMsg === 'string' ? errorMsg : JSON.stringify(errorMsg),
        timestamp: new Date().toISOString(),
      }
      setTerminalOutput(prev => [...prev, output])
      setShowTerminal(true)
    } finally {
      setIsExecuting(false)
    }
  }

  const handleTerminalCommand = async (e) => {
    if (e.key === 'Enter' && !isExecuting) {
      const command = terminalInput.trim()
      if (!command) return

      setTerminalOutput(prev => [...prev, {
        type: 'command',
        command,
        timestamp: new Date().toISOString(),
      }])

      setTerminalInput('')
      setIsExecuting(true)

      try {
        const response = await axios.post(`${API_BASE}/api/execute`, {
          command,
          language: 'bash',
        })

        const output = {
          type: 'output',
          stdout: String(response.data.stdout || ''),
          stderr: String(response.data.stderr || ''),
          success: response.data.success,
          timestamp: new Date().toISOString(),
        }

        setTerminalOutput(prev => [...prev, output])
      } catch (error) {
        const errorMsg = error.response?.data?.detail || error.response?.data?.message || error.message || 'Unknown error'
        const output = {
          type: 'error',
          error: typeof errorMsg === 'string' ? errorMsg : JSON.stringify(errorMsg),
          timestamp: new Date().toISOString(),
        }
        setTerminalOutput(prev => [...prev, output])
      } finally {
        setIsExecuting(false)
      }
    }
  }

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [chatMessages, currentResponse])

  const getLanguageFromPath = (path) => {
    const ext = path.split('.').pop()?.toLowerCase()
    const langMap = {
      js: 'javascript', jsx: 'javascript', ts: 'typescript', tsx: 'typescript',
      py: 'python', java: 'java', cpp: 'cpp', c: 'c', go: 'go', rs: 'rust',
      rb: 'ruby', php: 'php', html: 'html', css: 'css', json: 'json',
      yml: 'yaml', yaml: 'yaml', md: 'markdown'
    }
    return langMap[ext] || 'plaintext'
  }

  return (
    <div className="app-container">
      {/* Top Bar */}
      <div className="top-bar">
        <div className="top-bar-left">
          <Code size={20} />
          <h1>Better Cursor</h1>
        </div>
        <div className="top-bar-right">
          {activeTab && (
            <>
              <button
                className="run-button"
                onClick={() => {
                  const content = fileContents[activeTab] || ''
                  const mightNeedInput = content.includes('input(') || content.includes('input()')
                  if (mightNeedInput && !programInput && !needsInput) {
                    setNeedsInput(true)
                    setShowTerminal(true)
                    return
                  }
                  executeCode()
                }}
                disabled={isExecuting}
                title="Run code (Ctrl+R)"
              >
                <Play size={16} />
                Run
              </button>
              {needsInput && (
                <button
                  className="run-button"
                  onClick={() => {
                    setNeedsInput(false)
                    setProgramInput('')
                  }}
                  title="Disable input mode"
                  style={{ marginLeft: '8px', fontSize: '12px', padding: '4px 8px' }}
                >
                  Input: OFF
                </button>
              )}
            </>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        {/* File Explorer Sidebar */}
        <div className="file-explorer">
          <div className="explorer-header">
            <Folder size={16} />
            <span>Files</span>
            <button
              className="icon-btn"
              onClick={() => {
                const fileName = prompt('File name:')
                if (fileName) createFile(fileName)
              }}
              title="New File"
            >
              <Plus size={14} />
            </button>
          </div>
          <div className="file-list">
            {files.map((file) => (
              <div
                key={file.path}
                className={`file-item ${file.type}`}
                onClick={() => file.type === 'file' && openFile(file.path)}
                onContextMenu={(e) => {
                  e.preventDefault()
                  if (file.type === 'file' && confirm(`Delete ${file.path}?`)) {
                    deleteFile(file.path)
                  }
                }}
              >
                {file.type === 'directory' ? (
                  <Folder size={14} />
                ) : (
                  <File size={14} />
                )}
                <span>{file.name}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Editor Area */}
        <div className="editor-area">
          {/* File Tabs */}
          {openTabs.length > 0 && (
            <div className="file-tabs">
              {openTabs.map((tab) => (
                <div
                  key={tab}
                  className={`file-tab ${activeTab === tab ? 'active' : ''}`}
                  onClick={() => setActiveTab(tab)}
                >
                  <span>{tab.split('/').pop()}</span>
                  <button
                    className="tab-close"
                    onClick={(e) => {
                      e.stopPropagation()
                      closeTab(tab)
                    }}
                  >
                    <X size={12} />
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Monaco Editor */}
          {activeTab ? (
            <div className="editor-container">
              <Editor
                height="100%"
                language={getLanguageFromPath(activeTab)}
                value={fileContents[activeTab] || ''}
                onChange={handleEditorChange}
                theme="vs-dark"
                onMount={handleEditorDidMount}
                options={{
                  minimap: { enabled: true },
                  fontSize: 14,
                  wordWrap: 'on',
                  automaticLayout: true,
                  tabSize: 2,
                  scrollBeyondLastLine: false,
                }}
              />
            </div>
          ) : (
            <div className="empty-editor">
              <p>No file open. Select a file from the explorer or ask the LLM Council to create one.</p>
            </div>
          )}

          {/* Terminal */}
          {showTerminal && (
            <div className="terminal-panel">
              <div className="terminal-header">
                <span>Terminal</span>
                <button
                  className="icon-btn"
                  onClick={() => setShowTerminal(false)}
                >
                  <X size={14} />
                </button>
              </div>
              <div className="terminal-output">
                {terminalOutput.map((output, idx) => (
                  <div key={idx} className={`terminal-line ${output.type}`}>
                    {output.command && (
                      <div className="terminal-command">$ {String(output.command)}</div>
                    )}
                    {output.stdout && (
                      <div className="terminal-stdout">{String(output.stdout)}</div>
                    )}
                    {output.stderr && (
                      <div className="terminal-stderr">{String(output.stderr)}</div>
                    )}
                    {output.error && (
                      <div className="terminal-error">Error: {typeof output.error === 'string' ? output.error : JSON.stringify(output.error)}</div>
                    )}
                  </div>
                ))}
              </div>
              {/* Program Input Field (for programs that need input) */}
              {needsInput && (
                <div className="terminal-input-container" style={{ borderTop: '1px solid #333', padding: '8px', display: 'flex', alignItems: 'center', backgroundColor: '#1e1e1e', flexDirection: 'column' }}>
                  <div style={{ display: 'flex', alignItems: 'center', width: '100%', marginBottom: '4px' }}>
                    <span className="terminal-prompt" style={{ color: '#4CAF50', marginRight: '8px' }}>Input (one per line):</span>
                  </div>
                  <textarea
                    className="terminal-input"
                    value={programInput}
                    onChange={(e) => setProgramInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && !isExecuting) {
                        e.preventDefault()
                        executeCode(programInput || '')
                      }
                    }}
                    disabled={isExecuting}
                    placeholder="Enter input(s) for the program (one per line, press Cmd/Ctrl+Enter to run)...&#10;Example:&#10;username&#10;password&#10;yes"
                    style={{ 
                      flex: 1, 
                      width: '100%',
                      minHeight: '60px',
                      fontFamily: 'monospace',
                      fontSize: '12px',
                      padding: '4px',
                      resize: 'vertical'
                    }}
                    autoFocus
                  />
                  <div style={{ marginTop: '4px', fontSize: '11px', color: '#888' }}>
                    Press Cmd/Ctrl+Enter to run with input
                  </div>
                </div>
              )}
              <div className="terminal-input-container">
                <span className="terminal-prompt">$</span>
                <input
                  className="terminal-input"
                  value={terminalInput}
                  onChange={(e) => setTerminalInput(e.target.value)}
                  onKeyDown={handleTerminalCommand}
                  disabled={isExecuting}
                  placeholder="Enter command..."
                />
              </div>
            </div>
          )}
        </div>

        {/* Chat Panel */}
        {showChat && (
          <div className="chat-panel">
            <div className="chat-header">
              <div className="chat-header-left">
                <MessageSquare size={18} />
                <span>LLM Council Chat</span>
              </div>
              <button
                className="close-chat-btn"
                onClick={() => setShowChat(false)}
              >
                ×
              </button>
            </div>

            <div className="chat-messages">
              {chatMessages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`message ${msg.role} ${msg.isError ? 'error' : ''}`}
                >
                  <div className="message-content">
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>
                  {msg.fileWritten && (
                    <div className="file-written-notice">
                      ✓ Code written to: {msg.fileWritten}
                    </div>
                  )}
                  {msg.councilData && (
                    <div className="council-tabs">
                      <button
                        className={`tab ${councilTab === 'final' ? 'active' : ''}`}
                        onClick={() => setCouncilTab('final')}
                      >
                        Final Response
                      </button>
                      {msg.councilData.individual_responses.map((resp, i) => (
                        <button
                          key={i}
                          className={`tab ${councilTab === `model-${i}` ? 'active' : ''}`}
                          onClick={() => setCouncilTab(`model-${i}`)}
                        >
                          {resp.model.split('/').pop()}
                        </button>
                      ))}
                    </div>
                  )}
                  {msg.councilData && councilTab !== 'final' && (
                    <div className="council-response">
                      {(() => {
                        const tabIndex = parseInt(councilTab.split('-')[1])
                        const resp = msg.councilData.individual_responses[tabIndex]
                        return (
                          <div>
                            <div className="model-name">{resp.model}</div>
                            <ReactMarkdown>{resp.response}</ReactMarkdown>
                          </div>
                        )
                      })()}
                    </div>
                  )}
                </div>
              ))}
              {isLoading && (
                <div className="message assistant loading">
                  <Loader2 className="spinner" size={20} />
                  <span>LLM Council is thinking...</span>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            <div className="chat-input-container">
              <textarea
                className="chat-input"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                    e.preventDefault()
                    handleSendMessage()
                  }
                }}
                placeholder="Ask the LLM Council... (Cmd/Ctrl + Enter to send). Try: 'create a hello.py file' or 'write a function to hello.py'"
                rows={3}
              />
              <button
                className="send-button"
                onClick={handleSendMessage}
                disabled={!inputMessage.trim() || isLoading}
              >
                <Send size={18} />
              </button>
            </div>
          </div>
        )}

        {/* Chat Toggle Button */}
        {!showChat && (
          <button className="chat-toggle" onClick={() => setShowChat(true)}>
            <MessageSquare size={20} />
          </button>
        )}
      </div>
    </div>
  )
}

export default App
