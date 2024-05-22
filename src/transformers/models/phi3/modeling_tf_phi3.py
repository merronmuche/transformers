

import tensorflow as tf
import math
from tensorflow.keras import layers

class TFPhi3RMSNorm(tf.keras.layers.Layer):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super(TFPhi3RMSNorm, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.weight = self.add_weight(name='weight',
                                      shape=[hidden_size],
                                      initializer='ones',
                                      trainable=True)

    def call(self, inputs):
        input_dtype = inputs.dtype
        hidden_states = tf.cast(inputs, tf.float32)
        variance = tf.reduce_mean(tf.square(hidden_states), axis=-1, keepdims=True)
        normalized_states = hidden_states * tf.math.rsqrt(variance + self.variance_epsilon)
        return self.weight * tf.cast(normalized_states, input_dtype)

def _get_unpad_data(attention_mask):
    # Calculate sequence lengths in the batch
    seqlens_in_batch = tf.reduce_sum(attention_mask, axis=-1)

    # Get the indices of non-zero elements
    indices = tf.where(tf.reshape(attention_mask, [-1]))
    indices = tf.reshape(indices, [-1])

    # Find the maximum sequence length in the batch
    max_seqlen_in_batch = tf.reduce_max(seqlens_in_batch).numpy()

    # Compute cumulative sequence lengths
    cu_seqlens = tf.concat([[0], tf.cumsum(seqlens_in_batch)], axis=0)

    return indices, cu_seqlens, max_seqlen_in_batch

class TFPhi3RotaryEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, **kwargs):
        super(TFPhi3RotaryEmbedding, self).__init__(**kwargs)
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (tf.range(0, self.dim, 2, dtype=tf.float32) / self.dim))
        self.inv_freq = self.add_weight(name="inv_freq",
                                        shape=inv_freq.shape,
                                        initializer=tf.constant_initializer(inv_freq.numpy()),
                                        trainable=False)

    def call(self, x, position_ids):
        inv_freq_expanded = tf.expand_dims(self.inv_freq, axis=0)
        inv_freq_expanded = tf.expand_dims(inv_freq_expanded, axis=-1)
        inv_freq_expanded = tf.tile(inv_freq_expanded, [position_ids.shape[0], 1, 1])

        position_ids_expanded = tf.expand_dims(position_ids, axis=1)
        position_ids_expanded = tf.cast(position_ids_expanded, tf.float32)
        
        freqs = tf.matmul(inv_freq_expanded, position_ids_expanded)
        freqs = tf.transpose(freqs, perm=[0, 2, 1])
        emb = tf.concat([freqs, freqs], axis=-1)

        cos = tf.math.cos(emb)
        sin = tf.math.sin(emb)

        return tf.cast(cos, x.dtype), tf.cast(sin, x.dtype)

class TFPhi3SuScaledRotaryEmbedding(TFPhi3RotaryEmbedding):
    def __init__(self, dim, config, **kwargs):
        super(TFPhi3SuScaledRotaryEmbedding, self).__init__(dim, config.max_position_embeddings, config.rope_theta, **kwargs)
        self.short_factor = config.rope_scaling["short_factor"]
        self.long_factor = config.rope_scaling["long_factor"]
        self.original_max_position_embeddings = config.original_max_position_embeddings

    def call(self, x, position_ids, seq_len=None):
        seq_len = tf.reduce_max(position_ids) + 1
        if seq_len > self.original_max_position_embeddings:
            ext_factors = tf.constant(self.long_factor, dtype=tf.float32)
        else:
            ext_factors = tf.constant(self.short_factor, dtype=tf.float32)

        inv_freq_shape = tf.range(0, self.dim, 2, dtype=tf.float32) / self.dim
        self.inv_freq = 1.0 / (ext_factors * self.base**inv_freq_shape)

        inv_freq_expanded = tf.expand_dims(self.inv_freq, axis=0)
        inv_freq_expanded = tf.expand_dims(inv_freq_expanded, axis=-1)
        inv_freq_expanded = tf.tile(inv_freq_expanded, [position_ids.shape[0], 1, 1])

        position_ids_expanded = tf.expand_dims(position_ids, axis=1)
        position_ids_expanded = tf.cast(position_ids_expanded, tf.float32)

        freqs = tf.matmul(inv_freq_expanded, position_ids_expanded)
        freqs = tf.transpose(freqs, perm=[0, 2, 1])
        emb = tf.concat([freqs, freqs], axis=-1)

        scale = self.max_position_embeddings / self.original_max_position_embeddings
        if scale <= 1.0:
            scaling_factor = 1.0
        else:
            scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))

        cos = tf.math.cos(emb) * scaling_factor
        sin = tf.math.sin(emb) * scaling_factor

        return tf.cast(cos, x.dtype), tf.cast(sin, x.dtype)
    
class Phi3YarnScaledRotaryEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim, config, **kwargs):
        super(Phi3YarnScaledRotaryEmbedding, self).__init__(**kwargs)
        self.dim = dim
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta
        self.short_factor = config.rope_scaling["short_factor"]
        self.long_factor = config.rope_scaling["long_factor"]
        self.original_max_position_embeddings = config.original_max_position_embeddings

    def build(self, input_shape):
        super(Phi3YarnScaledRotaryEmbedding, self).build(input_shape)

    def call(self, x, position_ids, seq_len=None):
        seq_len = tf.reduce_max(position_ids) + 1
        if seq_len > self.original_max_position_embeddings:
            ext_factors = tf.convert_to_tensor(self.long_factor, dtype=tf.float32)
        else:
            ext_factors = tf.convert_to_tensor(self.short_factor, dtype=tf.float32)

        inv_freq_shape = tf.cast(tf.range(0, self.dim, 2), tf.float32) / self.dim
        inv_freq = 1.0 / (ext_factors * (self.base ** inv_freq_shape))

        inv_freq_expanded = tf.expand_dims(inv_freq, axis=0)
        inv_freq_expanded = tf.expand_dims(inv_freq_expanded, axis=2)
        inv_freq_expanded = tf.tile(inv_freq_expanded, [tf.shape(position_ids)[0], 1, 1])

        position_ids_expanded = tf.cast(tf.expand_dims(position_ids, axis=1), tf.float32)
        position_ids_expanded = tf.transpose(position_ids_expanded, perm=[0, 2, 1])


        # Perform the rotary embedding calculations
        freqs = tf.matmul(inv_freq_expanded, position_ids_expanded, transpose_b=True)
        emb = tf.concat([freqs, freqs], axis=-1)

        scale = self.max_position_embeddings / self.original_max_position_embeddings
        if scale <= 1.0:
            scaling_factor = 1.0
        else:
            scaling_factor = 0.1 * math.log(scale) + 1.0

        cos = tf.math.cos(emb) * scaling_factor
        sin = tf.math.sin(emb) * scaling_factor

        return tf.cast(cos, dtype=x.dtype), tf.cast(sin, dtype=x.dtype)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = tf.split(x, 2, axis=-1)
    return tf.concat([-x2, x1], axis=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`tf.Tensor`): The query tensor.
        k (`tf.Tensor`): The key tensor.
        cos (`tf.Tensor`): The cosine part of the rotary embedding.
        sin (`tf.Tensor`): The sine part of the rotary embedding.
        position_ids (`tf.Tensor`, optional): Deprecated and unused.
        unsqueeze_dim (`int`, optional, defaults to 1): The dimension to unsqueeze.

    Returns:
        `tuple(tf.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    if unsqueeze_dim == 1:
        cos = tf.expand_dims(cos, axis=1)
        sin = tf.expand_dims(sin, axis=1)
    elif unsqueeze_dim == 2:
        cos = tf.expand_dims(cos, axis=2)
        sin = tf.expand_dims(sin, axis=2)
    else:
        raise ValueError("unsqueeze_dim must be either 1 or 2")

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Phi3MLP(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(Phi3MLP, self).__init__(**kwargs)
        self.config = config
        self.gate_up_proj = tf.keras.layers.Dense(2 * config.intermediate_size, use_bias=False)
        self.down_proj = tf.keras.layers.Dense(config.hidden_size, use_bias=False)
        self.activation_fn = self.get_activation_function(config.hidden_act)

    def get_activation_function(self, activation_name):
        if activation_name == "relu":
            return tf.nn.relu
        elif activation_name == "gelu":
            return tf.nn.gelu
        elif activation_name == "tanh":
            return tf.nn.tanh
        elif activation_name == "sigmoid":
            return tf.nn.sigmoid
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}")

    def call(self, hidden_states):
        up_states = self.gate_up_proj(hidden_states)

        gate, up_states = tf.split(up_states, 2, axis=-1)
        up_states = up_states * self.activation_fn(gate)

        return self.down_proj(up_states)
    
def repeat_kv(hidden_states: tf.Tensor, n_rep: int) -> tf.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = tf.shape(hidden_states)
    if n_rep == 1:
        return hidden_states
    hidden_states = tf.expand_dims(hidden_states, axis=2)  # Shape: (batch, num_key_value_heads, 1, seqlen, head_dim)
    hidden_states = tf.tile(hidden_states, [1, 1, n_rep, 1, 1])  # Shape: (batch, num_key_value_heads, n_rep, seqlen, head_dim)
    return tf.reshape(hidden_states, [batch, num_key_value_heads * n_rep, slen, head_dim])

# class Phi3Attention(tf.keras.layers.Layer):
#     """Multi-headed attention from 'Attention Is All You Need' paper"""

#     def __init__(self, config, layer_idx=None, **kwargs):
#         super(Phi3Attention, self).__init__(**kwargs)
#         self.config = config
#         self.layer_idx = layer_idx

#         if layer_idx is None:
#             print(f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
#                   "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
#                   "when creating this class.")

#         self.attention_dropout = config.attention_dropout
#         self.hidden_size = config.hidden_size
#         self.num_heads = config.num_attention_heads
#         self.head_dim = self.hidden_size // self.num_heads
#         self.num_key_value_heads = config.num_key_value_heads
#         self.num_key_value_groups = self.num_heads // self.num_key_value_heads
#         self.max_position_embeddings = config.max_position_embeddings
#         self.original_max_position_embeddings = config.original_max_position_embeddings
#         self.rope_theta = config.rope_theta
#         self.rope_scaling = config.rope_scaling
#         self.is_causal = True

#         if (self.head_dim * self.num_heads) != self.hidden_size:
#             raise ValueError(
#                 f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
#                 f" and `num_heads`: {self.num_heads})."
#             )

#         op_size = self.num_heads * self.head_dim + 2 * (self.num_key_value_heads * self.head_dim)
#         self.o_proj = tf.keras.layers.Dense(self.hidden_size, use_bias=False)
#         self.qkv_proj = tf.keras.layers.Dense(op_size, use_bias=False)
#         self._init_rope()

#     def _init_rope(self):
#         if self.rope_scaling is None:
#             self.rotary_emb = TFPhi3RotaryEmbedding(
#                 self.head_dim,
#                 max_position_embeddings=self.max_position_embeddings,
#                 base=self.rope_theta,
#             )
#         else:
#             scaling_type = self.config.rope_scaling["type"]
#             if scaling_type == "su":
#                 self.rotary_emb = TFPhi3SuScaledRotaryEmbedding(self.head_dim, self.config)
#             elif scaling_type == "yarn":
#                 self.rotary_emb = Phi3YarnScaledRotaryEmbedding(self.head_dim, self.config)
#             else:
#                 raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

#     def call(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False):
#         print("You are not running the flash-attention implementation, expect numerical differences.")

#         bsz, q_len, _ = tf.shape(hidden_states)

#         qkv = self.qkv_proj(hidden_states)
#         query_pos = self.num_heads * self.head_dim
#         query_states = qkv[..., :query_pos]
#         key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
#         value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

#         query_states = tf.reshape(query_states, [bsz, q_len, self.num_heads, self.head_dim])
#         query_states = tf.transpose(query_states, [0, 2, 1, 3])
#         key_states = tf.reshape(key_states, [bsz, q_len, self.num_key_value_heads, self.head_dim])
#         key_states = tf.transpose(key_states, [0, 2, 1, 3])
#         value_states = tf.reshape(value_states, [bsz, q_len, self.num_key_value_heads, self.head_dim])
#         value_states = tf.transpose(value_states, [0, 2, 1, 3])

#         kv_seq_len = tf.shape(key_states)[-2]
#         if past_key_value is not None:
#             if self.layer_idx is None:
#                 raise ValueError(
#                     f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
#                     "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
#                     "with a layer index."
#                 )
#             kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
#         cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)

#         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

#         if past_key_value is not None:
#             cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
#             key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

#         # repeat k/v heads if n_kv_heads < n_heads
#         key_states = repeat_kv(key_states, self.num_key_value_groups)
#         value_states = repeat_kv(value_states, self.num_key_value_groups)

#         attn_weights = tf.matmul(query_states, key_states, transpose_b=True) / math.sqrt(self.head_dim)

#         if tf.shape(attn_weights) != (bsz, self.num_heads, q_len, kv_seq_len):
#             raise ValueError(
#                 f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
#                 f" {tf.shape(attn_weights)}"
#             )

#         if attention_mask is not None:
#             if tf.shape(attention_mask) != (bsz, 1, q_len, kv_seq_len):
#                 raise ValueError(
#                     f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {tf.shape(attention_mask)}"
#                 )
#             attn_weights += attention_mask

#         # upcast attention to fp32
#         attn_weights = tf.nn.softmax(attn_weights, axis=-1, dtype=tf.float32)
#         attn_weights = tf.nn.dropout(attn_weights, rate=self.attention_dropout)

#         attn_output = tf.matmul(attn_weights, value_states)

#         if tf.shape(attn_output) != (bsz, self.num_heads, q_len, self.head_dim):
#             raise ValueError(
#                 f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
#                 f" {tf.shape(attn_output)}"
#             )

#         attn_output = tf.transpose(attn_output, [0, 2, 1, 3])
#         attn_output = tf.reshape(attn_output, [bsz, q_len, self.hidden_size])

#         attn_output = self.o_proj(attn_output)

#         if not output_attentions:
#             attn_weights = None

#         return attn_output, attn_weights, past_key_value

class Phi3Attention(tf.keras.layers.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx=None):
        super(Phi3Attention, self).__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            tf.print(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.original_max_position_embeddings = config.original_max_position_embeddings
        self.rope_theta = config.rope_theta
        self.rope_scaling = config.rope_scaling
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        op_size = self.num_heads * self.head_dim + 2 * (self.num_key_value_heads * self.head_dim)
        self.o_proj = layers.Dense(self.hidden_size, use_bias=False)
        self.qkv_proj = layers.Dense(op_size, use_bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.rope_scaling is None:
            self.rotary_emb = TFPhi3RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            raise NotImplementedError("RoPE scaling not implemented in this example.")

    def call(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False):
        tf.print("You are not running the flash-attention implementation, expect numerical differences.")

        bsz, q_len, _ = tf.shape(hidden_states)

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        query_states = tf.transpose(tf.reshape(query_states, (bsz, q_len, self.num_heads, self.head_dim)), perm=[0, 2, 1, 3])
        key_states = tf.transpose(tf.reshape(key_states, (bsz, q_len, self.num_key_value_heads, self.head_dim)), perm=[0, 2, 1, 3])
        value_states = tf.transpose(tf.reshape(value_states, (bsz, q_len, self.num_key_value_heads, self.head_dim)), perm=[0, 2, 1, 3])

        kv_seq_len = tf.shape(key_states)[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = tf.matmul(query_states, key_states, transpose_b=True) / math.sqrt(self.head_dim)

        if tuple(tf.shape(attn_weights).numpy()) != (bsz.numpy(), self.num_heads, q_len.numpy(), kv_seq_len.numpy()):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {tf.shape(attn_weights)}"
            )

        if attention_mask is not None:
            if tuple(tf.shape(attention_mask).numpy()) != (bsz.numpy(), 1, q_len.numpy(), kv_seq_len.numpy()):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {tf.shape(attention_mask)}"
                )
            attn_weights = attn_weights + attention_mask

        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        attn_weights = tf.cast(attn_weights, dtype=value_states.dtype)
        attn_weights = tf.nn.dropout(attn_weights, rate=self.attention_dropout)

        attn_output = tf.matmul(attn_weights, value_states)

        if tuple(tf.shape(attn_output).numpy()) != (bsz.numpy(), self.num_heads, q_len.numpy(), self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {tf.shape(attn_output)}"
            )

        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, (bsz, q_len, self.hidden_size))

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


