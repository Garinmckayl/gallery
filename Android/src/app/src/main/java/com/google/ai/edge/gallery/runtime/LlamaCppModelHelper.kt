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

/**
 * llama.cpp backend for GGUF models (Bonsai 1-bit family).
 * Implements the same LlmModelHelper interface as the LiteRT backend,
 * so the Gallery UI works identically for both model formats.
 */
object LlamaCppModelHelper : LlmModelHelper {

  private const val TAG = "LlamaCppModelHelper"

  // Store SmolLM instance per model (keyed by model name)
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
        Log.i(TAG, "Model path: ${model.url}")

        val smolLM = SmolLM()

        val modelPath = model.getPath(context)
        Log.i(TAG, "Loading GGUF model from: $modelPath")

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

        // Add system prompt if provided
        val systemText = systemInstruction?.toString()
        if (!systemText.isNullOrEmpty()) {
          smolLM.addSystemPrompt(systemText)
        }

        instances[model.name] = smolLM
        model.instance = smolLM  // Store on model for later access

        Log.i(TAG, "llama.cpp model loaded: ${model.name}")
        launch(Dispatchers.Main) { onDone("") }

      } catch (e: Exception) {
        Log.e(TAG, "Failed to load model: ${model.name}", e)
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
    // SmolLM doesn't have a reset API -- we'd need to reload
    // For now, just log
    Log.i(TAG, "Reset conversation for ${model.name}")
  }

  override fun cleanUp(model: Model, onDone: () -> Unit) {
    try {
      val smolLM = instances.remove(model.name)
      smolLM?.close()
      model.instance = null
      Log.i(TAG, "Cleaned up ${model.name}")
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
        smolLM.getResponseAsFlow(input).collect { token ->
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
