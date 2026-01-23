[ ] Refactor du code
[ ] Implémenter une logique de jeu se basant sur les N meilleurs coups identifiés


à appronfondir : 

Option B : Attention (Squeeze-and-Excitation)


def _se_block(self, x, ratio=8):
    """Squeeze-and-Excitation pour attention sur les canaux"""
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([x, se])