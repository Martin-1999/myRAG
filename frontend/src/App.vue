<script setup>
import { onMounted, ref } from 'vue'
import axios from 'axios'

const api = axios.create({
  baseURL: 'http://127.0.0.1:8000/api'
})

const files = ref([])
const chunkMethod = ref('recursive')
const chunkSize = ref(800)
const chunkOverlap = ref(100)
const ingestResult = ref(null)
const ingestLoading = ref(false)
const ingestTask = ref(null)
let ingestPollTimer = null
const documents = ref([])
const documentsLoading = ref(false)
const deletingDocumentId = ref('')

const question = ref('')
const asking = ref(false)
const answerResult = ref(null)

const onFileChange = (event) => {
  files.value = Array.from(event.target.files || [])
}

const loadDocuments = async () => {
  documentsLoading.value = true
  try {
    const { data } = await api.get('/documents')
    documents.value = data.documents
  } finally {
    documentsLoading.value = false
  }
}

const stopIngestPolling = () => {
  if (ingestPollTimer) {
    clearInterval(ingestPollTimer)
    ingestPollTimer = null
  }
}

const pollIngestTask = async (taskId) => {
  const { data } = await api.get(`/ingest/tasks/${taskId}`)
  ingestTask.value = data
  if (data.status === 'completed') {
    ingestResult.value = data.result
    ingestLoading.value = false
    stopIngestPolling()
    await loadDocuments()
  } else if (data.status === 'failed') {
    ingestLoading.value = false
    stopIngestPolling()
    throw new Error(data.error || '入库失败')
  }
}

const ingest = async () => {
  if (!files.value.length) return
  ingestLoading.value = true
  ingestResult.value = null
  ingestTask.value = null
  try {
    const form = new FormData()
    files.value.forEach((file) => form.append('files', file))
    form.append('chunk_method', String(chunkMethod.value))
    form.append('chunk_size', String(chunkSize.value))
    form.append('chunk_overlap', String(chunkOverlap.value))
    const { data } = await api.post('/ingest/tasks', form, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    files.value = []
    await pollIngestTask(data.task_id)
    ingestPollTimer = setInterval(async () => {
      try {
        await pollIngestTask(data.task_id)
      } catch (error) {
        console.error(error)
      }
    }, 1500)
  } finally {
    if (ingestTask.value?.status !== 'running') {
      ingestLoading.value = false
    }
  }
}

const removeDocument = async (documentId) => {
  deletingDocumentId.value = documentId
  try {
    await api.delete(`/documents/${documentId}`)
    await loadDocuments()
  } finally {
    deletingDocumentId.value = ''
  }
}

const ask = async () => {
  if (!question.value.trim()) return
  asking.value = true
  try {
    const { data } = await api.post('/ask', {
      question: question.value,
      top_k: 5
    })
    answerResult.value = data
  } finally {
    asking.value = false
  }
}

onMounted(() => {
  loadDocuments()
})
</script>

<template>
  <div class="page">
    <header class="header">
      <div>
        <h1>RAG 本地应用</h1>
        <p>PDF 上传、建库、混合召回、重排与问答</p>
      </div>
    </header>

    <main class="layout">
      <section class="panel">
        <h2>文档入库</h2>
        <label class="field">
          <span>选择 PDF 文件</span>
          <input type="file" accept=".pdf" multiple @change="onFileChange" />
        </label>
        <div class="grid">
          <label class="field">
            <span>Chunk Method</span>
            <select v-model="chunkMethod">
              <option value="recursive">递归</option>
              <option value="fixed">固定大小</option>
            </select>
          </label>
          <label class="field">
            <span>Chunk Size</span>
            <input v-model.number="chunkSize" type="number" min="100" max="4000" />
          </label>
          <label class="field">
            <span>Chunk Overlap</span>
            <input v-model.number="chunkOverlap" type="number" min="0" max="1000" />
          </label>
        </div>
        <button class="primary" :disabled="ingestLoading" @click="ingest">
          {{ ingestLoading ? '处理中...' : '上传并建库' }}
        </button>

        <div v-if="ingestTask" class="result">
          <div class="section-head">
            <h3>入库进度</h3>
            <span class="status-text">{{ ingestTask.elapsed_seconds.toFixed(2) }}s</span>
          </div>
          <div class="progress-track">
            <div class="progress-bar" :style="{ width: `${ingestTask.progress}%` }"></div>
          </div>
          <p class="progress-line">
            {{ ingestTask.progress }}% · {{ ingestTask.current_step }} · {{ ingestTask.detail }}
          </p>
          <p class="progress-line">分块方式：{{ ingestTask.chunk_method === 'fixed' ? '固定大小' : '递归' }}</p>
          <p v-if="ingestTask.error" class="error-text">{{ ingestTask.error }}</p>
        </div>

        <div v-if="ingestResult" class="result">
          <h3>入库结果</h3>
          <p>文件数：{{ ingestResult.filenames.length }}</p>
          <p>Chunk 数：{{ ingestResult.chunk_count }}</p>
          <ul>
            <li v-for="name in ingestResult.filenames" :key="name">{{ name }}</li>
          </ul>
        </div>

        <div class="result">
          <div class="section-head">
            <h3>已入库文件</h3>
            <button class="secondary" :disabled="documentsLoading" @click="loadDocuments">
              {{ documentsLoading ? '刷新中...' : '刷新' }}
            </button>
          </div>
          <p v-if="!documents.length" class="empty">当前还没有已入库文件</p>
          <div v-for="item in documents" :key="item.document_id" class="document-row">
            <div class="document-meta">
              <strong>{{ item.filename }}</strong>
              <span>{{ item.chunk_count }} chunks</span>
              <span>{{ item.parser_backend }}</span>
            </div>
            <button
              class="danger"
              :disabled="deletingDocumentId === item.document_id"
              @click="removeDocument(item.document_id)"
            >
              {{ deletingDocumentId === item.document_id ? '删除中...' : '删除' }}
            </button>
          </div>
        </div>
      </section>

      <section class="panel">
        <h2>问答</h2>
        <label class="field">
          <span>输入问题</span>
          <textarea v-model="question" rows="5" placeholder="请输入你的问题"></textarea>
        </label>
        <button class="primary" :disabled="asking" @click="ask">
          {{ asking ? '生成中...' : '开始提问' }}
        </button>

        <div v-if="answerResult" class="result">
          <h3>回答</h3>
          <p class="answer">{{ answerResult.answer }}</p>
          <h3>完整提示词</h3>
          <pre class="prompt">{{ answerResult.prompt }}</pre>
          <h3>召回片段</h3>
          <div v-for="item in answerResult.retrieved_chunks" :key="item.chunk_id" class="chunk">
            <div class="chunk-meta">
              <strong>{{ item.metadata.filename }}</strong>
              <span>score={{ item.score.toFixed(4) }}</span>
              <span>{{ item.retrieval_sources.join(' + ') }}</span>
            </div>
            <p>{{ item.content }}</p>
          </div>
        </div>
      </section>
    </main>
  </div>
</template>
