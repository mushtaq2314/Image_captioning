# Image Captioning Model with Transformer and CNN Encoder

This project implements an image captioning model using a CNN-Transformer architecture. The model uses a CNN (InceptionV3) as an encoder to extract image features, followed by a Transformer-based encoder-decoder architecture to generate captions.

## Model Architecture

1. **CNN Encoder**: An InceptionV3 model (pre-trained on ImageNet) is used as the encoder. This model is frozen (not trained) and its output is reshaped to feed into the Transformer layers.
   
    ```python
    def CNN_Encoder():
        inception_v3 = tf.keras.applications.InceptionV3(
            include_top=False,
            weights='imagenet'
        )
        inception_v3.trainable = False

        output = inception_v3.output
        output = tf.keras.layers.Reshape(
            (-1, output.shape[-1]))(output)

        cnn_model = tf.keras.models.Model(inception_v3.input, output)
        return cnn_model
    ```

2. **Transformer Encoder Layer**: This layer consists of multi-head attention and dense layers with normalization and ReLU activation to encode the image features into embeddings.
   
    ```python
    class TransformerEncoderLayer(tf.keras.layers.Layer):
        def __init__(self, embed_dim, num_heads):
            super().__init__()
            self.layer_norm_1 = tf.keras.layers.LayerNormalization()
            self.layer_norm_2 = tf.keras.layers.LayerNormalization()
            self.attention = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim)
            self.dense = tf.keras.layers.Dense(embed_dim, activation="relu")

        def call(self, x, training):
            x = self.layer_norm_1(x)
            x = self.dense(x)
            attn_output = self.attention(query=x, value=x, key=x, training=training)
            x = self.layer_norm_2(x + attn_output)
            return x
    ```

3. **Embeddings**: Token embeddings and positional embeddings are used to prepare input tokens for the Transformer.
   
    ```python
    class Embeddings(tf.keras.layers.Layer):
        def __init__(self, vocab_size, embed_dim, max_len):
            super().__init__()
            self.token_embeddings = tf.keras.layers.Embedding(vocab_size, embed_dim)
            self.position_embeddings = tf.keras.layers.Embedding(max_len, embed_dim)

        def call(self, input_ids):
            length = tf.shape(input_ids)[-1]
            position_ids = tf.range(start=0, limit=length, delta=1)
            position_ids = tf.expand_dims(position_ids, axis=0)
            token_embeddings = self.token_embeddings(input_ids)
            position_embeddings = self.position_embeddings(position_ids)
            return token_embeddings + position_embeddings
    ```

4. **Transformer Decoder Layer**: The decoder layer includes multi-head attention and feed-forward networks. This is responsible for generating text captions from the image embeddings.

    ```python
    class TransformerDecoderLayer(tf.keras.layers.Layer):
        def __init__(self, embed_dim, units, num_heads):
            super().__init__()
            self.embedding = Embeddings(vocab_size, embed_dim, max_len)
            self.attention_1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
            self.attention_2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
            self.ffn_layer_1 = tf.keras.layers.Dense(units, activation="relu")
            self.ffn_layer_2 = tf.keras.layers.Dense(embed_dim)
            self.out = tf.keras.layers.Dense(vocab_size, activation="softmax")

        def call(self, input_ids, encoder_output, training, mask=None):
            embeddings = self.embedding(input_ids)
            attn_output_1 = self.attention_1(query=embeddings, value=embeddings, key=embeddings, training=training)
            out_1 = self.layernorm_1(embeddings + attn_output_1)
            attn_output_2 = self.attention_2(query=out_1, value=encoder_output, key=encoder_output, training=training)
            out_2 = self.layernorm_2(out_1 + attn_output_2)
            ffn_out = self.ffn_layer_1(out_2)
            ffn_out = self.ffn_layer_2(ffn_out)
            ffn_out = self.layernorm_3(ffn_out + out_2)
            preds = self.out(ffn_out)
            return preds
    ```

5. **Training the Model**: The model is trained with sparse categorical cross-entropy loss and early stopping for efficient training.

    ```python
    caption_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, clipvalue=1.0),
        loss=cross_entropy
    )
    history = caption_model.fit(
        train_dataset,
        epochs=50,
        validation_data=val_dataset,
        callbacks=[early_stopping]
    )
    ```

## Installation

1. Clone this repository.
   ```bash
   git clone https://github.com/yourusername/image-captioning-transformer.git
   cd image-captioning-transformer
