
<template>
  <div class="chat-container"> 
    
    <aside class="sidebar"> 
      <div class="sidebar-header">
        <h3>对话历史</h3>
        
        <button @click="newChat" class="new-chat-btn">+ 新对话</button>
      </div>

      
      <div class="chat-list">

        
        <div
          v-for="chat in chatHistory"
          :key="chat.id"
          :class="['chat-item', { active: chat.id === currentSessionId }]"
          @click="switchChat(chat.id)"
        >
          
          <span class="chat-title">{{ chat.title }}</span>
          <span class="chat-time">{{ formatTime(chat.updatedAt) }}</span>

          
          <button class="chat-delete-btn" title="删除该对话" 
                  @click.stop="deleteChat(chat.id)" >×</button>
        </div>
      </div>
      
      <div class="session-id-bar" title="当前会话 ID">当前会话：{{ currentSessionId }}</div>
    </aside>

    
    <main class="chat-main"> 
      
      <div class="messages" ref="messagesContainer">
        
        <div v-for="msg in messages" :key="msg.id":class="['message', msg.role]">
          
          <div class="message-avatar">{{ msg.role === 'user' ? '我' : 'AI' }}</div>
          
          
          <div class="message-content">
            
            <div v-html="renderMarkdown(msg.content)" class="markdown-body"></div>

            
            <div v-if="msg.images && msg.images.length" class="msg-images">
              <img
                v-for="(im, i) in msg.images"
                :key="i"
                :src="im.url"
                :alt="im.name"
                class="msg-img"
              />
            </div>
            
            
            <div v-if="msg.toolSteps && msg.toolSteps.length" class="toolSteps">
              
              <details>
                <summary>推理过程 ({{ msg.toolSteps.length }}步)</summary>
                <div class="toolStepList">
                  
                  <div v-for="(step, i) in msg.toolSteps" :key="i" class="toolStep">
                    <span class="step-round">第{{ step.round }}轮</span>
                    
                    <span :class="['step-type', step.step]">{{ getStepLabel(step.step) }}</span>
                    
                    <span v-if="step.tool" class="step-tool">{{ step.tool }}</span>
                    <pre v-if="step.content" class="step-args">{{ step.content }}</pre>
                    <pre v-if="step.args" class="step-args">{{ formatArgs(step.args) }}</pre>
                    <pre v-if="step.result" class="step-result">{{ step.result }}</pre>
                  </div>
                </div>
              </details>
            </div>

            
            
            <div v-if="msg.batchSummary" class="mm-summary">
              <div class="mm-summary-title">📊 批量缺陷统计</div>
              <div v-html="renderMarkdown(msg.batchSummary)" class="markdown-body"></div>
            </div>
             
            <div v-if="msg.multimodalResults && msg.multimodalResults.length" class="mm-results">
              <div v-for="(r, i) in msg.multimodalResults" :key="i" class="mm-result-card">
                <div class="mm-result-header">图片 {{ (r.img_index ?? i) + 1 }}：{{ r.filename }}</div>
                >
                <div v-if="r.image_description" class="mm-desc">📝 {{ r.image_description }}</div>
                <div v-html="renderMarkdown(r.answer)" class="markdown-body"></div>
                <div v-if="r.toolSteps && r.toolSteps.length" class="toolSteps">
                  <details>
                    <summary>推理过程 ({{ r.toolSteps.length }}步)</summary>
                    <div class="toolStepList">
                      <div v-for="(step, j) in r.toolSteps" :key="j" class="toolStep">
                        <span class="step-round">第{{ step.round }}轮</span>
                        <span :class="['step-type', step.step]">{{ getStepLabel(step.step) }}</span>
                        <span v-if="step.tool" class="step-tool">{{ step.tool }}</span>
                        <pre v-if="step.content" class="step-args">{{ step.content }}</pre>
                        <pre v-if="step.args" class="step-args">{{ formatArgs(step.args) }}</pre>
                        <pre v-if="step.result" class="step-result">{{ step.result }}</pre>
                      </div>
                    </div>
                  </details>
                </div>
              </div>
            </div>
          </div>
        </div>

        
        <div v-if="isLoading" class="message ai">
          <div class="message-avatar">AI</div>
          <div class="message-content">
            <div class="typing-indicator">
              <span></span><span></span><span></span>
            </div>
          </div>
        </div>
      </div>

      
      <div class="input-area">
        <div class="input-wrapper">
          
          <input
            ref="fileInput"
            type="file"
            multiple
            accept="image/*"
            class="file-input-hidden"
            @change="onFileSelected" 
          />
          >
          <button class="img-btn" title="上传图片（可多选）" @click="fileInput.click()" :disabled="isLoading">📎</button>
          >
          
          
          <div v-if="selectedImages.length" class="img-preview-list">
            <div v-for="(img, i) in selectedImages" :key="i" class="img-preview-item">
              <img :src="img.url" :alt="img.name" />
              >
              <span class="img-remove" @click="removeImage(i)">×</span>
            </div>
          </div>

          
          <textarea
            v-model="inputText"
            @keydown.enter.exact.prevent="sendMessage"
            placeholder="输入您的问题，Shift+Enter换行..."
            rows="2"
            :disabled="isLoading"
          ></textarea>

          
          <label class="stream-toggle">
            <input type="checkbox" v-model="streamMode" />
            流式输出
          </label>

          
          <button @click="sendMessage" :disabled="isLoading || (!inputText.trim() && !selectedImages.length)" class="send-btn">
  发送
          </button>
        </div>
        <div class="input-hint">工业质检智能问答 | 支持图文/多图上传、流式输出 </div>
      </div>
    </main>
  </div>
</template> 

<script setup>

import { ref, nextTick, onMounted } from 'vue'

import axios from 'axios'
import { marked } from 'marked'

const API_BASE = 'http://localhost:8000/api'

const messages = ref([])
const inputText = ref('')
const isLoading = ref(false)
const currentSessionId = ref('')
const chatHistory = ref([])
const messagesContainer = ref(null)
let abortController = null
let msgIdCounter = 0

const fileInput = ref(null)
const selectedImages = ref([])
const streamMode = ref(true)

onMounted(() => {
  newChat()            
  loadChatHistory()    
})

function genMsgId() {
  return `msg_${Date.now()}_${++msgIdCounter}`
}

function newChat() {
  
  if (abortController) {
    abortController.abort()
    abortController = null
  }
  
  currentSessionId.value = 'session_' + Date.now()
  
  messages.value = []
  inputText.value = ''
  isLoading.value = false
}

function loadChatHistory() {
  try {
    const saved = localStorage.getItem('chat_history')
    if (saved) {
      chatHistory.value = JSON.parse(saved)
    }
  } catch (e) {
    
    console.warn('对话历史解析失败，已重置：', e)
    chatHistory.value = []
    localStorage.removeItem('chat_history')
  }
}

function saveChatHistory() {
  localStorage.setItem('chat_history', JSON.stringify(chatHistory.value))
}

function switchChat(sessionId) {
  
  if (sessionId === currentSessionId.value) return

  
  if (abortController) {
    abortController.abort()
    abortController = null
  }

  isLoading.value = false
  currentSessionId.value = sessionId

  
  const chat = chatHistory.value.find(c => c.id === sessionId)
  if (chat) {
    
    
    messages.value = chat.messages || []
  }
}

async function deleteChat(sessionId) {
  
  
  
  const chat = chatHistory.value.find(c => c.id === sessionId)
  
  
  if (!chat) return
  
  
  
  
  if (!confirm(`确定删除对话「${chat.title}」吗？此操作不可恢复。`)) return

  
  
  
  chatHistory.value = chatHistory.value.filter(c => c.id !== sessionId)
  
  
  
  saveChatHistory()

  
  try {
    
    
    
    await fetch(`${API_BASE}/session/${encodeURIComponent(sessionId)}`, { method: 'DELETE' })
  } catch (e) {
    
    
    console.warn('后端会话删除失败（不影响前端删除）：', e)
  }

  
  
  
  if (sessionId === currentSessionId.value) {
    
    
    if (chatHistory.value.length > 0) {
      
      
      switchChat(chatHistory.value[0].id)
    } else {
      
      
      newChat()
    }
  }
  
}

function updateChatHistory(firstQuestion) {
  
  
  
  const existing = chatHistory.value.find(c => c.id === currentSessionId.value)

  if (existing) {
    

    
    
    
    existing.messages = [...messages.value]

    
    existing.updatedAt = Date.now()
  } else {
    
    

    chatHistory.value.unshift({
      
      id: currentSessionId.value,

      
      
      title: firstQuestion.slice(0, 20) + (firstQuestion.length > 20 ? '...' : ''),

      
      messages: [...messages.value],

      
      createdAt: Date.now(),
      updatedAt: Date.now()
    })
  }
  saveChatHistory()
}

async function sendMessage() {
  const question = inputText.value.trim()
  
  const hasImages = selectedImages.value.length > 0
  
  const useStream = streamMode.value
  
  if ((!question && !hasImages) || isLoading.value) return

  
  
  
  const msgList = messages.value
  const prevMsgLen = msgList.length
  const userImages = selectedImages.value.map(im => ({ name: im.name, url: im.url }))
  msgList.push({ id: genMsgId(), role: 'user', content: question, images: userImages })
  
  const imgsToSend = selectedImages.value
  selectedImages.value = []
  inputText.value = ''
  
  isLoading.value = true

  
  msgList.push({ id: genMsgId(), role: 'ai', content: '', toolSteps: [], multimodalResults: [], batchSummary: '' })
  const aiMessage = msgList[msgList.length - 1]
  await scrollToBottom()
  let toolRound = 0  
  abortController = new AbortController()

  
  
  const formData = new FormData()
  formData.append('question', question)
  formData.append('session_id', currentSessionId.value)
  formData.append('stream', useStream ? 'true' : 'false')
  imgsToSend.forEach((im) => formData.append('images', im.file))
  
  try {
    
    const response = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      body: formData,  
      signal: abortController.signal
    })
    if (!response.ok) {
      throw new Error(`服务器返回错误：${response.status} ${response.statusText}`)
    }
    if (useStream) {
      
      const reader = response.body.getReader()
      

      const decoder = new TextDecoder()
      let buffer = ''  
      while (true) {
        const { done, value } = await reader.read()
        
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        
        
        
        
        
        const rawEvents = buffer.split('\n\n')  
        buffer = rawEvents.pop()  
        
        for (const rawEvent of rawEvents) {
          if (!rawEvent.startsWith('data: ')) continue
          const data = JSON.parse(rawEvent.slice(6))
          if (data.type === 'token') {
            aiMessage.content += data.content
          } else if (data.type === 'tool_call') {
            toolRound++
            aiMessage.toolSteps.push({ round: toolRound, step: 'action', tool: data.tool, args: data.args })
          } else if (data.type === 'tool_result') {
            aiMessage.toolSteps.push({ round: toolRound, step: 'observation', tool: data.tool, result: data.result })
          } else if (data.type === 'image_done') {
            // 多模态 SSE：逐张图片结果到达即渲染
            aiMessage.multimodalResults.push({
              img_index: data.img_index,
              filename: data.filename || '未知',
              image_description: data.image_description || '',
              answer: data.answer || data.error || '（无回答）',
              toolSteps: data.tool_steps || []
            })
            if (data.answer && !aiMessage.content) aiMessage.content = data.answer
          } else if (data.type === 'batch_summary') {
            aiMessage.batchSummary = data.batchSummary || ''
            if (aiMessage.batchSummary) aiMessage.content = aiMessage.batchSummary
          } else if (data.type === 'done') {
            if (aiMessage.multimodalResults.length) {
              aiMessage.multimodalResults.sort((a, b) => (a.img_index ?? 0) - (b.img_index ?? 0))
            }
            aiMessage.content = data.answer || aiMessage.content
            updateChatHistory(question)
          } else if (data.type === 'error') {
            aiMessage.content = `抱歉，发生错误：${data.message}`
          }
          
          await scrollToBottom()
        }
      }
    } else {
      
      const data = await response.json()
      
      if (data.mode === 'multimodal') {
        
        aiMessage.multimodalResults = (data.multimodalResults || []).map((r) => ({
          img_index: r.img_index,
          filename: r.filename || '未知',
          image_description: r.image_description || '',
          answer: r.answer || r.error || '（无回答）',
          toolSteps: r.tool_steps || []
        }))
        
        aiMessage.batchSummary = data.batchSummary || ''
        
        if (aiMessage.batchSummary) {
          aiMessage.content = aiMessage.batchSummary
        } else if (aiMessage.multimodalResults.length && aiMessage.multimodalResults[0].answer) {
          aiMessage.content = aiMessage.multimodalResults[0].answer
        }
      } else {
        
        aiMessage.content = data.answer || '（无回答）'
        if (Array.isArray(data.tool_steps)) {
          aiMessage.toolSteps = data.tool_steps.map((s) => ({
            round: s.round,
            step: s.step === 'thought' ? 'reasoning' : s.step,
            
            
            tool: s.tool,
            args: s.args,
            result: s.result,
            content: s.content
          }))
        }
      }
      updateChatHistory(question)
    }
  } catch (error) {
    if (error.name === 'AbortError') {
      
      msgList.splice(prevMsgLen)
     
      return
    }
    aiMessage.content = `抱歉，网络错误：${error.message}`
  } finally {
    isLoading.value = false     
    abortController = null      
    await scrollToBottom()      
  }
}

function onFileSelected(event) {
  
  const files = Array.from(event.target.files || [])
  
  files.forEach((f) => {
    if (!f.type.startsWith('image/')) return  
    selectedImages.value.push({ file: f, name: f.name, url: URL.createObjectURL(f) })
    
    
    
  })
  event.target.value = ''  
}

function removeImage(index) {
  const im = selectedImages.value[index]
  if (im && im.url) URL.revokeObjectURL(im.url)  
  
  selectedImages.value.splice(index, 1)
}

function renderMarkdown(text) {
  return marked.parse(text || '')
}

function getStepLabel(step) {
  const labels = {
    reasoning: '思考',       
    action: '调用工具',      
    observation: '返回结果'  
  }
  return labels[step] || step  
}

function formatArgs(args) {
  if (typeof args === 'string') return args
  try {
    
    return JSON.stringify(args, null, 2)
  } catch {
    return String(args)
  }
}

function formatTime(timestamp) {
  const date = new Date(timestamp)
  
  
  const pad = n => String(n).padStart(2, '0')
  return `${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`
}

async function scrollToBottom() {
  await nextTick()
  if (messagesContainer.value) {
    
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    
  }
}

</script>

<style scoped>

.chat-container {
  display: flex;
  height: 100vh;  
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  
}

.sidebar {
  width: 260px;              
  background: #f7f7f8;       
  border-right: 1px solid #e5e5e5;  
  display: flex;
  flex-direction: column;    
}

.sidebar-header {
  padding: 16px;

  border-bottom: 1px solid #e5e5e5;
}

.sidebar-header h3 {
  margin: 0;

  font-size: 16px;
}

.new-chat-btn {
  width: 100%;
  padding: 8px;
  background: #10a37f;       
  color: white;
  border: none;
  border-radius: 6px;        
  
  cursor: pointer;           
  margin-top: 8px;
}

.new-chat-btn:hover { background: #0d8c6d; }

.chat-list {
  flex: 1;                   
  
  overflow-y: auto;          
  padding: 8px;
}

.session-id-bar {
  padding: 8px 16px;
  font-size: 11px;
  color: #999;
  border-top: 1px solid #037e6a;
  white-space: nowrap;
  overflow: hidden;  
  text-overflow:ellipsis;
}

.chat-item {
  position: relative;  
  padding: 10px 12px;  
  border-radius: 6px;  
  cursor: pointer;     
  margin-bottom: 4px;  
}

.chat-item:hover {
  background: #ececf1; 
}

.chat-item.active {
  background: #d9d9e3; 
}

.chat-title {
  display: block;           
  font-size: 13px;          
  white-space: nowrap;      
  overflow: hidden;         
  text-overflow: ellipsis;  
}

.chat-time {
  display: block;     
  font-size: 11px;    
  color: #999;        
  margin-top: 2px;    
}

.chat-delete-btn {
  position: absolute;  
  top: 8px;            
  right: 8px;          
  display: none;       
  border: none;        
  background: transparent; 
  color: #999;         
  font-size: 16px;     
  line-height: 1;      
  cursor: pointer;     
  padding: 2px 5px;    
  border-radius: 4px;  
}

.chat-item:hover .chat-delete-btn {
  display: inline-block; 
}

.chat-delete-btn:hover {
  background: #e0e0e6; 
  color: #d33;          
}

.chat-main {
  flex: 1;                   
  display: flex;
  flex-direction: column;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
}

.message {
  display: flex;
  gap: 16px;                 
  margin-bottom: 24px;
  max-width: 800px;          
  margin-left: auto;         
  margin-right: auto;
}

.message.user { flex-direction: row-reverse; }

.message-avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;        
  display: flex;
  align-items: center;       
  justify-content: center;   
  font-size: 14px;
  font-weight: 600;
  flex-shrink: 0;            
}

.message.user .message-avatar { background: #10a37f; color: white; }
.message.ai .message-avatar { background: #19c37d; color: white; }

.message-content {
  flex: 1;
  line-height: 1.6;          
}

.markdown-body :deep(p) { margin: 0 0 10px; }
.markdown-body :deep(pre) {
  background: #f6f8fa;
  padding: 12px;
  border-radius: 6px;
  overflow-x: auto;          
  font-size: 13px;
}
.markdown-body :deep(code) { font-family: 'Consolas', monospace; }
.markdown-body :deep(h1), .markdown-body :deep(h2), .markdown-body :deep(h3) {
  margin: 16px 0 8px;
}

.markdown-body :deep(ul), .markdown-body :deep(ol) { padding-left: 20px; }

.markdown-body :deep(table) { border-collapse: collapse; width: 100%; }

.markdown-body :deep(th), .markdown-body :deep(td) {
  border: 1px solid #ddd;
  padding: 6px 12px;
  text-align: left;
}

.toolSteps { margin-top: 12px; }

.toolSteps details {
  background: #f7f7f8;
  padding: 8px 12px;
  border-radius: 6px;
}
.toolSteps summary {
  cursor: pointer;
  font-size: 13px;
  color: #666;
}
.toolStep { padding: 4px 0; font-size: 13px; }
.step-round { color: #666; margin-right: 8px; }
.step-type { font-weight: 600; margin-right: 8px; }

.step-type.action { color: #d69e2e; }        
.step-type.observation { color: #3182ce; }   
.step-tool { color: #333; font-family: monospace; }
.step-args {
  background: #fff;
  padding: 6px 8px;
  margin: 4px 0;
  border-radius: 4px;
  font-size: 12px;
  color: #555;
  white-space: pre-wrap;                     
  border-left: 3px solid #d69e2e;            
}
.step-result {
  background: #fff;
  padding: 8px;
  margin-top: 4px;
  border-radius: 4px;
  font-size: 12px;
  white-space: pre-wrap;
  border-left: 3px solid #3182ce;            
}

.input-area {
  border-top: 1px solid #e5e5e5;
  padding: 16px;
  background: white;
}

.input-wrapper {
  display: flex;
  gap: 12px;
  
  max-width: 800px;
  margin: 0 auto;
}

textarea {
  flex: 1;
  padding: 12px;
  border: 1px solid #e5e5e5;
  border-radius: 8px;
  resize: none;              
  font-size: 14px;
  outline: none;             
  font-family: inherit;
}

textarea:focus { border-color: #10a37f; }

.send-btn {
  padding: 0 24px;
  background: #10a37f;
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 14px;
}

.send-btn:hover:not(:disabled) { background: #0d8c6d; }

.send-btn:disabled { background: #ccc; cursor: not-allowed; }

.input-hint {
  text-align: center;
  color: #999;
  font-size: 12px;
  margin-top: 8px;
}

.typing-indicator { display: flex; gap: 4px; padding: 8px 0; }
.typing-indicator span {
  width: 8px;
  height: 8px;
  background: #ccc;
  border-radius: 50%;
  animation: typing 1.4s infinite;
}

.typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.4s; }

@keyframes typing {
  0%, 60%, 100% { transform: translateY(0); }   
  
  30% { transform: translateY(-8px); }            
}

.file-input-hidden { display: none; }      

.img-btn {
  padding: 0 10px;
  background: #f0f0f0;
  border: 1px solid #e5e5e5;
  border-radius: 8px;
  cursor: pointer;
  font-size: 16px;
}
.img-btn:hover:not(:disabled) { background: #e2e2e2; }
.img-btn:disabled { opacity: 0.5; cursor: not-allowed; }

.input-wrapper { flex-wrap: wrap; }  
.img-preview-list {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  flex-basis: 100%;  
}
.img-preview-item { position: relative; }
.img-preview-item img {
  width: 48px;
  height: 48px;
  object-fit: cover;
  border-radius: 6px;
  border: 1px solid #e5e5e5;
  display: block;
}

.img-remove {
  position: absolute;
  top: -6px;
  right: -6px;
  width: 16px;
  height: 16px;
  line-height: 14px;
  text-align: center;
  background: #e53e3e;
  color: #fff;
  border-radius: 50%;
  font-size: 12px;
  cursor: pointer;
}

.msg-images { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
.msg-img {
  max-width: 200px;
  max-height: 200px;
  border-radius: 8px;
  border: 1px solid #e5e5e5;
}

.stream-toggle {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  color: #666;
  white-space: nowrap;
  cursor: pointer;
  user-select: none;
}
.stream-toggle-disabled { opacity: 0.5; cursor: not-allowed; }

.mm-results { display: flex; flex-direction: column;
   gap: 12px; margin-top: 8px; }
.mm-result-card {
  border: 1px solid #e5e5e5;
  border-radius: 8px;
  padding: 12px;
  background: #fafafa;
}
.mm-result-header { font-size: 13px; font-weight: 600;
   color: #333; margin-bottom: 6px; }
.mm-desc {
  font-size: 13px;
  color: #666;
  background: #f0f4ff;
  border-left: 3px solid #3182ce;
  padding: 6px 8px;
  border-radius: 4px;
  margin-bottom: 8px;
}

.mm-summary {
  border: 1px solid #e5e5e5;
  border-radius: 8px;
  padding: 12px;
  background: #f0fdf4;
  margin-top: 8px;
  margin-bottom: 8px;
}
.mm-summary-title { font-size: 13px; font-weight: 600; color: #333; margin-bottom: 6px; }

</style>

