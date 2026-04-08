package com.google.ai.edge.gallery.runtime

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.gallery.data.ConfigKeys
import com.google.ai.edge.gallery.data.Model
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ToolProvider
import io.shubham0204.smollm.SmolLM
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch

private const val IVY_SYSTEM_PROMPT = """You are Ivy, an AI tutor for Ethiopian students. Respond in English unless the student writes in Amharic. Use the Socratic method: ask guiding questions, never give answers directly. Be warm and encouraging. Keep responses to 2-3 sentences. Use Ethiopian examples when relevant."""

object LlamaCppModelHelper : LlmModelHelper {

  private const val TAG = "LlamaCppModelHelper"
  private val instances = mutableMapOf<String, SmolLM>()
  private var currentJob: Job? = null

  override fun initialize(
    context: Context,
    model: Model,
    supportImage: Boolean,
    supportAudio: Boolean,
    onDone: (String) -> Unit,
    systemInstruction: Contents?,
    tools: List<ToolProvider>,
    enableConversationConstrainedDecoding: Boolean,
    coroutineScope: CoroutineScope?,
  ) {
    val scope = coroutineScope ?: CoroutineScope(Dispatchers.IO)
    scope.launch {
      try {
        Log.i(TAG, "Initializing llama.cpp for: ${model.name}")

        val smolLM = SmolLM()
        val modelPath = model.getPath(context)
        Log.i(TAG, "Loading GGUF from: $modelPath")

        smolLM.load(
          modelPath = modelPath,
          params = SmolLM.InferenceParams(
            contextSize = 1024,
            temperature = model.getFloatConfigValue(ConfigKeys.TEMPERATURE, 0.7f),
            minP = 0.1f,
            useMmap = true,
            useMlock = false,
            storeChats = false,
          )
        )

        // Inject Ivy's Socratic tutor system prompt directly
        smolLM.addSystemPrompt(IVY_SYSTEM_PROMPT)
        Log.i(TAG, "System prompt injected (${IVY_SYSTEM_PROMPT.length} chars)")

        instances[model.name] = smolLM
        model.instance = smolLM

        Log.i(TAG, "Model loaded: ${model.name}")
        launch(Dispatchers.Main) { onDone("") }

      } catch (e: Exception) {
        Log.e(TAG, "Failed to load: ${model.name}", e)
        launch(Dispatchers.Main) { onDone("Error: ${e.message}") }
      }
    }
  }

  override fun resetConversation(
    model: Model,
    supportImage: Boolean,
    supportAudio: Boolean,
    systemInstruction: Contents?,
    tools: List<ToolProvider>,
    enableConversationConstrainedDecoding: Boolean,
  ) {
    Log.i(TAG, "Reset conversation for ${model.name}")
  }

  override fun cleanUp(model: Model, onDone: () -> Unit) {
    try {
      val smolLM = instances.remove(model.name)
      smolLM?.close()
      model.instance = null
    } catch (e: Exception) {
      Log.e(TAG, "Cleanup error", e)
    }
    onDone()
  }

  override fun runInference(
    model: Model,
    input: String,
    resultListener: ResultListener,
    cleanUpListener: CleanUpListener,
    onError: (message: String) -> Unit,
    images: List<Bitmap>,
    audioClips: List<ByteArray>,
    coroutineScope: CoroutineScope?,
    extraContext: Map<String, String>?,
  ) {
    val smolLM = instances[model.name] ?: (model.instance as? SmolLM)
    if (smolLM == null) {
      onError("Model not initialized")
      return
    }

    val scope = coroutineScope ?: CoroutineScope(Dispatchers.IO)
    currentJob = scope.launch {
      try {
        var hasTokens = false
        smolLM.getResponseAsFlow(input).collect { token ->
          hasTokens = true
          launch(Dispatchers.Main) {
            resultListener(token, false, null)
          }
        }
        launch(Dispatchers.Main) {
          resultListener("", true, null)
          cleanUpListener()
        }
      } catch (e: Exception) {
        Log.e(TAG, "Inference error", e)
        launch(Dispatchers.Main) {
          onError(e.message ?: "Inference failed")
          cleanUpListener()
        }
      }
    }
  }

  override fun stopResponse(model: Model) {
    currentJob?.cancel()
    currentJob = null
  }
}
