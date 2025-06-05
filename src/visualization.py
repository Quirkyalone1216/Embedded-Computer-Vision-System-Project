import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
import numpy as np
import os

# 新增：繪製訓練過程中的損失與正確率曲線，並存檔到 results 資料夾
def plot_training_curves(history):
    h = history.history
    epochs = range(1, len(h['loss']) + 1)

    # 繪製總 Loss 曲線
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, h['loss'],  label='train loss')
    plt.plot(epochs, h['val_loss'], label='val loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/output/training_val_loss.png')
    plt.close()

    # 年齡分支 Accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, h['predications_age_accuracy'],  label='train age acc')
    plt.plot(epochs, h['val_predications_age_accuracy'], label='val age acc')
    plt.title('Age Branch Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/output/age_accuracy.png')
    plt.close()

    # 性別分支 Accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, h['predications_gender_accuracy'],  label='train gender acc')
    plt.plot(epochs, h['val_predications_gender_accuracy'], label='val gender acc')
    plt.title('Gender Branch Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/output/gender_accuracy.png')
    plt.close()

    # 人種分支 Accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, h['predications_race_accuracy'],  label='train race acc')
    plt.plot(epochs, h['val_predications_race_accuracy'], label='val race acc')
    plt.title('Race Branch Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/output/race_accuracy.png')
    plt.close()


# 新增：在測試集上計算並呈現各任務的最終指標，並輸出 CSV 及混淆矩陣圖片到 results
def evaluate_on_testset(model, x_test, y_test_a, y_test_g, y_test_r):
    # 評估 Loss 與 Accuracy
    results = model.evaluate(
        x_test, [y_test_a, y_test_g, y_test_r],
        batch_size=32, verbose=0
    )
    total_loss, age_loss, gender_loss, race_loss, age_acc, gender_acc, race_acc = (
        results[0], results[1], results[2], results[3], results[4], results[5], results[6]
    )

    # 取得預測結果
    preds = model.predict(x_test, batch_size=32, verbose=0)
    pred_age    = np.argmax(preds[0], axis=1)
    pred_gender = np.argmax(preds[1], axis=1)
    pred_race   = np.argmax(preds[2], axis=1)

    true_age    = np.argmax(y_test_a, axis=1)
    true_gender = np.argmax(y_test_g, axis=1)
    true_race   = np.argmax(y_test_r, axis=1)

    # 建立字典來儲存最終指標，並存成 JSON
    metrics = {
        'age':    {'loss': age_loss,    'accuracy': age_acc},
        'gender': {'loss': gender_loss, 'accuracy': gender_acc},
        'race':   {'loss': race_loss,   'accuracy': race_acc}
    }

    # 性別 & 人種的混淆矩陣
    cm_gender = confusion_matrix(true_gender, pred_gender)
    cm_race   = confusion_matrix(true_race, pred_race)

    # 繪製並存檔混淆矩陣的 Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_gender,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Male', 'Female'],
        yticklabels=['Male', 'Female']
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix: Gender')
    plt.tight_layout()
    plt.savefig('results/cm_gender.png')
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_race,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['White','Black','Asian','Indian','Others'],
        yticklabels=['White','Black','Asian','Indian','Others']
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix: Race')
    plt.tight_layout()
    plt.savefig('results/cm_race.png')
    plt.close()

    # 將 classification report 也輸出成 CSV
    cr_age    = classification_report(true_age, pred_age, output_dict=True)
    cr_gender = classification_report(true_gender, pred_gender, output_dict=True)
    cr_race   = classification_report(true_race, pred_race, output_dict=True)

    print("已將測試結果儲存至 results 資料夾")


if __name__ == '__main__':
    os.makedirs('results/output', exist_ok=True)
    # 1. 從 history.json 讀取訓練記錄並執行繪圖
    with open('results/history.json', 'r', encoding='utf-8') as f:
        hist_dict = json.load(f)
    class DummyHistory:
        def __init__(self, history_dict):
            self.history = history_dict
    history = DummyHistory(hist_dict)
    plot_training_curves(history)

    # 2. 從 test_metrics.json 讀取測試指標並分別產生 Loss 與 Accuracy 長條圖
    with open('results/test_metrics.json', 'r', encoding='utf-8') as f:
        test_metrics = json.load(f)
    branches = ['age', 'gender', 'race']
    losses = [test_metrics[b]['loss'] for b in branches]
    accuracies = [test_metrics[b]['accuracy'] for b in branches]
    x = np.arange(len(branches))
    bar_width = 0.6
    # 繪製 Test Loss 長條圖
    plt.figure(figsize=(6, 4))
    plt.bar(x, losses, width=bar_width, label='Test Loss', alpha=0.7)
    plt.xticks(x, branches)
    plt.title('Test Loss by Branch')
    plt.xlabel('Branch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/output/test_loss_by_branch.png')
    plt.close()
    # 繪製 Test Accuracy 長條圖
    plt.figure(figsize=(6, 4))
    plt.bar(x, accuracies, width=bar_width, label='Test Accuracy', alpha=0.7)
    plt.xticks(x, branches)
    plt.title('Test Accuracy by Branch')
    plt.xlabel('Branch')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/output/test_accuracy_by_branch.png')
    plt.close()

    # 3. 讀取並儲存各分類報表為 CSV
    for branch in ['age', 'gender', 'race']:
        with open(f'results/cr_{branch}.json', 'r', encoding='utf-8') as f:
            cr = json.load(f)
        # 將 classification report 直接以 JSON 形式儲存
        with open(f'results/output/cr_{branch}_report.json', 'w', encoding='utf-8') as jf:
            json.dump(cr, jf, ensure_ascii=False, indent=2)

    print("已完成所有圖表與 CSV 的輸出。")
