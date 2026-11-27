# Déploiement de l'API sur Azure Container Apps

Cible : abonnement **Azure for Students**, région **France Central** (régions autorisées :
francecentral, switzerlandnorth, spaincentral, italynorth, austriaeast).
L'image (sans données) est publiée sur **ghcr.io** par le workflow `Déploiement API` ;
au démarrage, l'API télécharge l'index `champion` depuis MLflow (DagsHub).

## Coût et garde-fous

| Réglage | Valeur | Pourquoi |
|---|---|---|
| Plan | Consumption seul (`--enable-workload-profiles false`) | Pas de profil Dedicated : il déclenche un compteur de gestion ≈ 72 $/mois |
| Répliques | min 0, max 1 | S'éteint sans trafic : couvert par l'offre gratuite mensuelle (≈ 100 h actives à 0,5 vCPU / 1 Gio) |
| Ressources | 0,5 vCPU, 1 Gio | Mesuré : 153 Mio en mémoire, démarrage ≈ 26 s |
| Journaux | `--logs-destination none` | Pas d'espace Log Analytics facturé (le flux en direct reste disponible) |
| Registre | ghcr.io public | Pas d'Azure Container Registry (≈ 5 $/mois) |
| À éviter | fenêtre de maintenance, private endpoint | Déclenchent aussi le compteur de gestion |

Pour la soutenance : `--min-replicas 1` pendant la démo (évite le démarrage à froid),
puis retour à `0`. Laisser 1 réplique en permanence coûterait ≈ 10 $/mois.

## 1. Préalables (une fois)

1. Ajouter dans `.env` une clé d'administration aléatoire :
   ```powershell
   uv run python -c "import secrets; print('ADMIN_API_KEY=' + secrets.token_urlsafe(32))"
   ```
2. Lancer une première fois le workflow **Déploiement API** (onglet Actions) : il publie
   l'image `ghcr.io/obiyohan/puls-events-api` (l'étape Azure est sautée).
3. Rendre l'image publique : GitHub → profil → **Packages** → `puls-events-api` →
   **Package settings** → **Change visibility** → **Public**. L'image ne contient aucun secret.

## 2. Ressources Azure (PowerShell)

```powershell
$RG = "rg-puls-events"; $LOC = "francecentral"
$ENVN = "cae-puls-events"; $APP = "puls-events-api"
$IMG = "ghcr.io/obiyohan/puls-events-api:latest"

az provider register --namespace Microsoft.ManagedIdentity
az extension add --name containerapp --upgrade

# Environnement Consumption, sans journaux stockés
az containerapp env create -n $ENVN -g $RG -l $LOC --logs-destination none --enable-workload-profiles false

# Secrets lus dans .env (jamais affichés)
$e = @{}
Get-Content .env | Where-Object { $_ -match '^\s*([A-Z_]+)=(.*)$' } | ForEach-Object { $e[$Matches[1]] = $Matches[2].Trim().Trim('"') }

az containerapp create -n $APP -g $RG --environment $ENVN --image $IMG `
  --target-port 8000 --ingress external --min-replicas 0 --max-replicas 1 --cpu 0.5 --memory 1.0Gi `
  --secrets "azure-openai-key=$($e.AZURE_OPENAI_API_KEY)" "mlflow-password=$($e.MLFLOW_TRACKING_PASSWORD)" "admin-key=$($e.ADMIN_API_KEY)" `
  --env-vars "AZURE_OPENAI_ENDPOINT=$($e.AZURE_OPENAI_ENDPOINT)" "AZURE_OPENAI_API_KEY=secretref:azure-openai-key" `
             "MLFLOW_TRACKING_URI=$($e.MLFLOW_TRACKING_URI)" "MLFLOW_TRACKING_USERNAME=$($e.MLFLOW_TRACKING_USERNAME)" `
             "MLFLOW_TRACKING_PASSWORD=secretref:mlflow-password" "ADMIN_API_KEY=secretref:admin-key" `
  --query properties.configuration.ingress.fqdn -o tsv
```

La dernière commande affiche l'adresse publique (`puls-events-api.<…>.francecentral.azurecontainerapps.io`).

## 3. Identité de déploiement pour GitHub Actions (OIDC, sans mot de passe)

Identité managée affectée par l'utilisateur (aucune inscription d'application requise),
limitée au rôle *Contributor* sur la seule Container App.

```powershell
az identity create -n id-github-deploy -g $RG -l $LOC
$CLIENT = az identity show -n id-github-deploy -g $RG --query clientId -o tsv
$PRINCIPAL = az identity show -n id-github-deploy -g $RG --query principalId -o tsv
$APP_ID = az containerapp show -n $APP -g $RG --query id -o tsv
az role assignment create --assignee-object-id $PRINCIPAL --assignee-principal-type ServicePrincipal --role Contributor --scope $APP_ID
az identity federated-credential create --name github-master --identity-name id-github-deploy -g $RG `
  --issuer https://token.actions.githubusercontent.com `
  --subject repo:ObiYohan/puls-events-llmops:ref:refs/heads/master --audiences api://AzureADTokenExchange
"AZURE_CLIENT_ID=$CLIENT"; "AZURE_TENANT_ID=$(az account show --query tenantId -o tsv)"; "AZURE_SUBSCRIPTION_ID=$(az account show --query id -o tsv)"
```

## 4. Secrets et variables GitHub (Settings → Secrets and variables → Actions)

| Type | Nom | Valeur |
|---|---|---|
| Secret | `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID` | Affichés à l'étape 3 (identifiants, pas des mots de passe) |
| Secret | `ADMIN_API_KEY` | Même valeur que dans `.env` (rechargement après promotion) |
| Variable | `AZURE_RESOURCE_GROUP` | `rg-puls-events` |
| Variable | `AZURE_CONTAINERAPP_NAME` | `puls-events-api` |
| Variable | `API_URL` | `https://<fqdn affiché à l'étape 2>` |

Ensuite, chaque push modifiant l'API redéploie l'image, et chaque promotion du pipeline
hebdomadaire appelle `/reload`.

## Vérifier

```powershell
$URL = "https://" + (az containerapp show -n $APP -g $RG --query properties.configuration.ingress.fqdn -o tsv)
curl.exe -s "$URL/health"          # premier appel ≈ 30 s si l'application était à l'arrêt
az containerapp logs show -n $APP -g $RG --tail 20
```
