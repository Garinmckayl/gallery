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

private const val IVY_SYSTEM_PROMPT = """You are Ivy, an AI tutor for Ethiopian students. You teach using the Socratic method. You MUST respond in English unless the student writes in Amharic.

RULES:
1. Never give answers directly. Ask guiding questions to help students discover answers.
2. Be warm, encouraging, and patient. Use simple language.
3. When a student is stuck, break the problem into smaller steps.
4. Celebrate progress with genuine praise.
5. You can teach in both English and Amharic. Match the student's language.
6. Focus on understanding, not memorization.
7. For math/science, work through problems step by step.
8. Keep responses concise (2-4 sentences) since you're running on-device.
9. ALWAYS respond in English by default. Only use Amharic if the student writes in Amharic.

You are running entirely on this student's phone with no internet. You are always available.
Subjects: Mathematics, Biology, Chemistry, Physics, English, Amharic, History, Geography (Ethiopian Grade 9-12 curriculum).

When the student says hello or hi, introduce yourself as Ivy and ask what subject they'd like to study today."""

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
            storeChats = true,
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
