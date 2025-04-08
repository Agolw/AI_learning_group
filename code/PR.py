import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, average_precision_score, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC

def pr_0():
    precision = [0.8, 0.7, 0.6, 0.5]
    recall = [0.2, 0.4, 0.6, 0.8]

    # 绘制PR曲线
    plt.plot(recall, precision, '-.')

    # 添加标题和轴标签
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    # 显示图形
    plt.show()

def pr_1():
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
    y_scores = np.array([0.2, 0.8, 0.3, 0.7, 0.9, 0.1, 0.6, 0.4, 0.85, 0.95])
    # 计算精确率、召回率和阈值
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    # 绘制PR曲线
    plt.plot(recall, precision)

    # 添加标题和轴标签
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    # 显示图形
    plt.show()

def pr_2():
    # 模拟数据（假设已训练模型）
    y_true = [0, 1, 1, 0, 1, 1, 0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.35, 0.8, 0.7, 0.6, 0.2, 0.3, 0.9, 0.5]

    # 通过precisioin_recall_curve获得precison, recall, thresholds
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    print(precision, recall, thresholds)
    print(ap)

    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve (AP={ap:.2f})')
    plt.show()

def pr_3():
    # 加载糖尿病数据集
    data = datasets.load_breast_cancer()
    X = data.data
    y = data.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 训练逻辑回归模型
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 预测概率
    y_scores = model.predict_proba(X_test)[:, 1]

    # 计算精确率和召回率
    precision, recall, _ = precision_recall_curve(y_test, y_scores)

    # 计算PR曲线下面积
    pr_auc = auc(recall, precision)

    # 绘制PR图
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Diabetes Classification')
    plt.legend(loc='best')
    plt.show()

def pr_4():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification

    # 生成样本数据
    X, y = make_classification(n_samples=1000, n_features=20, weights=[0.9, 0.1], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 训练模型
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_scores = clf.predict_proba(X_test)[:, 1]

    # 计算 Precision-Recall 曲线
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

    # 绘制 PR 曲线
    plt.figure(figsize=(8,6))
    plt.plot(recall, precision, marker='.', label='PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid()
    plt.show()

def pr_5():
    # 加载糖尿病数据集
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target

    # 将目标值转换为二分类问题
    # 根据中位数划分高进展和低进展两类
    median_target = np.median(y)
    y_binary = (y > median_target).astype(int)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

    # 训练逻辑回归模型
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 预测概率
    y_scores = model.predict_proba(X_test)[:, 1]

    # 计算精确率和召回率
    precision, recall, _ = precision_recall_curve(y_test, y_scores)

    # 计算PR曲线下面积
    pr_auc = auc(recall, precision)

    # 绘制PR图
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Diabetes Classification')
    plt.legend(loc='best')
    plt.show()

def roc_0():
    # 设定假正率（False Positive Rate）的值
    fpr = [0.1, 0.2, 0.3, 0.4]
    # 设定真正率（True Positive Rate）的值
    tpr = [0.2, 0.4, 0.6, 0.8]

    # 绘制 ROC 曲线
    plt.plot(fpr, tpr, 'k-.^')

    # 绘制随机猜测的线（对角线）
    plt.plot([0, 1], [0, 1], 'k--')

    # 添加标题和轴标签
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # 设置坐标轴范围
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    # 显示图形
    plt.show()

def roc_1():
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
    y_scores = np.array([0.2, 0.8, 0.3, 0.7, 0.9, 0.1, 0.6, 0.4, 0.85, 0.95])
    # 计算精确率、召回率和阈值
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # 绘制PR曲线
    plt.plot(fpr, tpr, linestyle='-', marker='.')
    plt.plot([0, 1], [0, 1], 'k--')

    # 添加标题和轴标签
    plt.title('ROC Curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')

    # 显示图形
    plt.show()

def roc_2():
    # 模拟数据（假设已训练模型）
    y_true = [0, 1, 1, 0, 1, 1, 0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.35, 0.8, 0.7, 0.6, 0.2, 0.3, 0.9, 0.5]

    # 通过precisioin_recall_curve获得precison, recall, thresholds
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    print(precision, recall, thresholds)
    print(ap)

    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve (AP={ap:.2f})')
    plt.show()

def roc_3():
    y_true = [0, 1, 1, 0, 1, 1, 0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.35, 0.8, 0.7, 0.6, 0.2, 0.3, 0.9, 0.5]
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    print(fpr)
    print(tpr)
    print(thresholds)
    auc = roc_auc_score(y_true, y_scores)

    plt.plot(fpr, tpr, marker='.')
    plt.plot([0, 1], [0, 1], linestyle='--')  # 随机猜测线
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC={auc:.2f})')
    plt.show()

def roc_4():
    # 生成模拟二分类数据
    X, y = datasets.make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练逻辑回归模型
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 预测概率
    y_scores = model.predict_proba(X_test)[:, 1]

    # 计算 FPR、TPR 和阈值
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    # 绘制 ROC 曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def roc_5():
    # 加载鸢尾花数据集
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # 转换为二分类问题（只考虑前两类）
    X = X[y != 2]
    y = y[y != 2]
    y = label_binarize(y, classes=[0, 1]).ravel()

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练逻辑回归模型
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 预测概率
    y_scores = model.predict_proba(X_test)[:, 1]

    # 计算 FPR、TPR 和阈值
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    # 绘制 ROC 曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for Iris dataset (binary classification)')
    plt.legend(loc="lower right")
    plt.show()

def roc_6():
    # 生成复杂的二分类数据集
    X, y = datasets.make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5,
                            n_clusters_per_class=2, random_state=42)

    # 初始化分类器
    classifiers = [
        LogisticRegression(max_iter=1000),
        SVC(kernel='rbf', probability=True)
    ]

    # 初始化交叉验证对象
    cv = StratifiedKFold(n_splits=5)

    # 为每个分类器创建存储 ROC 信息的列表
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # 遍历每个分类器
    for classifier in classifiers:
        # 存储每个折叠的 TPR 和 AUC
        tprs_fold = []
        aucs_fold = []
        # 进行交叉验证
        for train, test in cv.split(X, y):
            # 训练模型
            classifier.fit(X[train], y[train])
            # 预测概率
            if hasattr(classifier, "decision_function"):
                probas_ = classifier.decision_function(X[test])
            else:
                probas_ = classifier.predict_proba(X[test])[:, 1]
            # 计算 ROC 曲线
            fpr, tpr, thresholds = roc_curve(y[test], probas_)
            # 插值计算 TPR 在平均 FPR 上的值
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs_fold.append(interp_tpr)
            # 计算 AUC
            roc_auc = auc(fpr, tpr)
            aucs_fold.append(roc_auc)

        # 计算平均 TPR 和 AUC
        mean_tpr = np.mean(tprs_fold, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs_fold)

        # 存储平均 TPR 和 AUC
        tprs.append(mean_tpr)
        aucs.append(mean_auc)

        # 绘制平均 ROC 曲线和标准差范围
        plt.plot(mean_fpr, mean_tpr,
                label=r'%s (AUC = %0.2f $\pm$ %0.2f)' % (classifier.__class__.__name__, mean_auc, std_auc),
                lw=2, alpha=.8)

        # 绘制标准差范围
        std_tpr = np.std(tprs_fold, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)

    # 绘制随机猜测的对角线
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    # 设置图形属性
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example with cross - validation')
    plt.legend(loc="lower right")

    # 保存图形
    plt.savefig('complex_roc_curve.png')
    # 显示图形
    plt.show()

def cc_1():
    # 模拟真正率（TPR）和假正率（FPR）
    tpr = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    fpr = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    # 不同的正类先验概率
    priors = np.linspace(0, 1, 100)

    # 初始化代价曲线数据
    costs = []

    # 计算不同先验概率下的代价
    for prior in priors:
        cost = prior * (1 - tpr) + (1 - prior) * fpr
        costs.append(cost)

    costs = np.array(costs)
    # 绘制代价曲线
    plt.figure()
    for i in range(len(tpr)):
        plt.plot(priors, costs[:, i], label=f'Threshold {i+1}')

    plt.xlabel('Positive Class Prior Probability')
    plt.ylabel('Expected Cost')
    plt.title('Cost Curve')
    plt.legend()
    plt.show()

def cc_2():
    # 生成模拟二分类数据
    X, y = datasets.make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练逻辑回归模型
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 预测概率
    y_scores = model.predict_proba(X_test)[:, 1]

    # 计算 FPR 和 TPR
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)

    # 不同的正类先验概率
    priors = np.linspace(0, 1, 100)

    # 初始化代价曲线数据
    costs = []

    # 计算不同先验概率下的代价
    for prior in priors:
        cost = prior * (1 - tpr) + (1 - prior) * fpr
        costs.append(cost)
    costs = np.array(costs)
    # 绘制代价曲线
    plt.figure()
    for i in range(len(thresholds)):
        plt.plot(priors, costs[:, i], label=f'Threshold {i+1}')

    plt.xlabel('Positive Class Prior Probability')
    plt.ylabel('Expected Cost')
    plt.title('Cost Curve')
    plt.legend()
    plt.show()

def cc_3():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve
    from sklearn.preprocessing import label_binarize

    # 加载鸢尾花数据集
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # 转换为二分类问题（只考虑前两类）
    X = X[y != 2]
    y = y[y != 2]
    y = label_binarize(y, classes=[0, 1]).ravel()

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练逻辑回归模型
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 预测概率
    y_scores = model.predict_proba(X_test)[:, 1]

    # 计算 FPR 和 TPR
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)

    # 不同的正类先验概率
    priors = np.linspace(0, 1, 100)

    # 初始化代价曲线数据
    costs = []

    # 计算不同先验概率下的代价
    for prior in priors:
        cost = prior * (1 - tpr) + (1 - prior) * fpr
        costs.append(cost)
    costs = np.array(costs)
    # 绘制代价曲线
    plt.figure()
    for i in range(len(thresholds)):
        plt.plot(priors, costs[:, i], label=f'Threshold {i+1}')

    plt.xlabel('Positive Class Prior Probability')
    plt.ylabel('Expected Cost')
    plt.title('Cost Curve')
    plt.legend()
    plt.show()

def cc_4():
    # 加载乳腺癌数据集
    data = datasets.load_breast_cancer()
    X = data.data
    y = data.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练逻辑回归模型
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 预测概率
    y_scores = model.predict_proba(X_test)[:, 1]

    # 计算假正率（FPR）和真正率（TPR）
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)

    # 生成一系列正类先验概率
    priors = np.linspace(0, 1, 100)

    # 初始化代价曲线数据
    costs = []

    # 计算不同先验概率下的代价
    for prior in priors:
        cost = prior * (1 - tpr) + (1 - prior) * fpr
        costs.append(cost)
    costs = np.array(costs)
    # 绘制代价曲线
    plt.figure()
    for i in range(len(thresholds)):
        plt.plot(priors, costs[:, i], label=f'Threshold {i + 1}')

    plt.xlabel('Positive Class Prior Probability')
    plt.ylabel('Expected Cost')
    plt.title('Cost Curve for Breast Cancer Classification')
    plt.legend()
    plt.show()
    

if __name__ == '__main__':
    cc_2()