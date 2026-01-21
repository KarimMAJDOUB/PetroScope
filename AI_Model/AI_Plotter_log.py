import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def save_performance_plot(df_results, model_name="Model"):
    """
    Sauvegarde les graphiques dans le dossier : AI_Model/AI_plots.
    Ajout des messages d'erreur sans modifier le code original.
    """
    if df_results is None or df_results.empty:
        print("Plotter : Pas de données à dessiner.")
        return

    try:
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            current_dir = os.getcwd()
            print("Info : '__file__' non défini, utilisation du répertoire courant.")

        plots_dir = os.path.join(current_dir, "AI_plots")

        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
    except PermissionError as e:
        print(f"Erreur création dossier AI_plots : Permission refusée ({e})")
        return
    except OSError as e:
        print(f"Erreur création dossier AI_plots :")
        return
    except Exception as e:
        print(f"Erreur création dossier AI_plots :")
        return

    if 'DAYTIME' not in df_results.columns:
        print("Erreur : la colonne 'DAYTIME' est absente du DataFrame.")
        return

    try:
        df_results['DAYTIME'] = pd.to_datetime(df_results['DAYTIME'])
    except Exception as e:
        print(f"Erreur conversion DAYTIME en datetime :")
        return

    x_axis = df_results['DAYTIME']

    fluids_config = {
        'Huile': {
            'real': ['BORE_OIL_VOL', 'OIL_VOL_test'],  # RF vs LSTM
            'pred': ['OIL_VOL', 'OIL_VOL_pred']        # RF vs LSTM
        },
        'Gaz': {
            'real': ['BORE_GAS_VOL', 'GAS_VOL_test'],
            'pred': ['GAS_VOL', 'GAS_VOL_pred']
        },
        'Eau': {
            'real': ['BORE_WAT_VOL', 'WAT_VOL_test'],
            'pred': ['WAT_VOL', 'WAT_VOL_pred']
        }
    }

    timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%M")
    print(f"Génération des graphiques dans {plots_dir}...")

    for fluid_name, config in fluids_config.items():
        col_real = next((c for c in df_results.columns if c in config['real']), None)
        col_pred = next((c for c in df_results.columns if c in config['pred']), None)

        if col_pred and col_real:
            try:
                plt.figure(figsize=(12, 6))

                # Courbe Réelle (Bleu)
                plt.plot(x_axis, df_results[col_real], label='Réel', color='blue', linewidth=2)

                # Courbe Prédiction (Rouge pointillé)
                plt.plot(x_axis, df_results[col_pred], label=f'Prédiction ({model_name})',
                         color='red', linestyle='--', linewidth=2)

                plt.title(f"{model_name} - {fluid_name}", fontsize=14)
                plt.xlabel("Date (DAYTIME)")
                plt.ylabel("Volume")
                plt.legend()
                plt.grid(True, linestyle=':', alpha=0.6)

                filename = f"{model_name}_{fluid_name}_{timestamp}.png"
                plt.savefig(os.path.join(plots_dir, filename))
                plt.close()
                print(f"Image générée : {filename}")
            except PermissionError as e:
                print(f"Erreur sauvegarde {fluid_name} : Permission refusée ({e})")
            except OSError as e:
                print(f"Erreur sauvegarde {fluid_name} : OS error ({e})")
            except ValueError as e:
                print(f"Erreur graphique {fluid_name} : ValueError ({e})")
            except TypeError as e:
                print(f"Erreur graphique {fluid_name} : TypeError ({e})")
            except Exception as e:
                print(f"Erreur inattendue pour {fluid_name} : {e}")
        else:
            if not col_real:
                print(f"Aucune colonne réelle trouvée pour {fluid_name}.")
            if not col_pred:
                print(f"Aucune colonne prédite trouvée pour {fluid_name}.")
