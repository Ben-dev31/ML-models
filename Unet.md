# Implémentation d’un U-Net en PyTorch
Le U-Net est un réseau de neurones convolutionnel utilisé pour la segmentation d’images, notamment biomédicales.
Son architecture en forme de “U” combine :

* une **partie descendante (encodeur)** qui extrait les caractéristiques,

* une **partie montante (decodeur)** qui reconstruit une carte de segmentation à la taille originale,
avec des **connexions de saut** (skip connections) pour préserver les détails fins.

##  Dépendance 
* PyTorch

## Impléméntation 
* **Bloc de convolution double**
    ```python
    class DoubleConv(nn.Module)
    ```
💡 **Explication :**

Chaque bloc du U-Net contient deux convolutions 3×3 suivies d’une normalisation de lot (BatchNorm) et d’une activation ReLU.

* padding=1 conserve la taille spatiale.

* inplace=True évite de créer des copies mémoire inutiles.

👉 Ce bloc apprend des motifs à différentes échelles tout en stabilisant l’entraînement.

## Partie descendante : Encodeur
```python 
class UNet(nn.Module)
```
💡 **Explication :**

Chaque étape :

1. applique une double convolution pour extraire des caractéristiques,

2. puis réduit la taille de moitié via le MaxPool2d.

👉 L’image devient plus petite mais plus riche en informations.

## Le “bottleneck” (fond du U)
```python
self.bottleneck = DoubleConv(512, 1024)
```
C’est le niveau le **plus profond** du réseau, où les représentations sont très compressées et abstraites.
Il relie la partie descendante à la partie montante.

## Partie montante : Décodeur
💡 **Explication :**

* Chaque ``ConvTranspose2d`` double la taille spatiale (upsampling appris).

* On **concatène** les caractéristiques montantes avec celles de la descente correspondante (skip connections).

* Puis on applique un ``DoubleConv`` pour combiner ces informations.

## Sortie finale
```python
self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

```
Une convolution 1×1 ramène les canaux à la dimension de sortie souhaitée :

* 1 canal → segmentation binaire

* N canaux → segmentation multi-classes

## Exemple d’utilisation
```python
if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1)
    x = torch.randn(1, 3, 256, 256)  # (batch, channels, H, W)
    y = model(x)
    print("Sortie :", y.shape)

```
## Entraînement du modèle (binaire)
### Définir la fonction de perte et l’optimiseur
```python 
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

```

### Boucle d’entraînement simplifiée
```python 
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, masks)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

```
## 📈 Résumé de l’architecture
| Étape         | Opération            | Taille typique (pour 256×256) |
| ------------- | -------------------- | ----------------------------- |
| Entrée        | Image RGB            | 256×256×3                     |
| Conv1         | DoubleConv(3→64)     | 256×256×64                    |
| Pool1         | MaxPool              | 128×128×64                    |
| Conv2         | DoubleConv(64→128)   | 128×128×128                   |
| Pool2         | MaxPool              | 64×64×128                     |
| Conv3         | DoubleConv(128→256)  | 64×64×256                     |
| Pool3         | MaxPool              | 32×32×256                     |
| Conv4         | DoubleConv(256→512)  | 32×32×512                     |
| Bottleneck    | DoubleConv(512→1024) | 16×16×1024                    |
| Up + concat   | (x4 + up(1024→512))  | 32×32×1024                    |
| …             | …                    | …                             |
| Sortie finale | Conv1x1(64→1)        | 256×256×1                     |

