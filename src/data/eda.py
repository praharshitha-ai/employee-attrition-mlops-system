'''Basic EDA summary (STEP 3).'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use('seaborn-v0_8')
os.makedirs('experiments/eda', exist_ok=True)

def eda():
    df = pd.read_csv('data/raw/attrition.csv')
    
    print("Dataset shape:", df.shape)
    print("\nTarget balance:")
    print(df['Attrition'].value_counts(normalize=True))
    
    print("\nMissing values:")
    print(df.isnull().sum().sum())
    
    print("\nNumeric features describe:")
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('EmployeeNumber')
    print(df[num_cols].describe())
    
    print("\nCategorical features:")
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        print(f"{col}: {df[col].value_counts()}")
    
    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Attrition by Age
    df.boxplot(column='Age', by='Attrition', ax=axes[0,0])
    axes[0,0].set_title('Age by Attrition')
    
    # Monthly Income
    df.boxplot(column='MonthlyIncome', by='Attrition', ax=axes[0,1])
    axes[0,1].set_title('Monthly Income by Attrition')
    
    # Attrition rate by Department
    df.groupby('Department')['Attrition'].value_counts(normalize=True).unstack().plot(kind='bar', ax=axes[1,0])
    axes[1,0].set_title('Attrition by Department')
    
    # Correlation heatmap (numeric)
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
    axes[1,1].set_title('Numeric Features Correlation')
    
    plt.tight_layout()
    plt.savefig('experiments/eda/eda_summary.png')
    plt.show()
    
    print("EDA plots saved to experiments/eda/")
    
    # Feature importance baseline (Random Forest)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    X = pd.get_dummies(df.drop('Attrition', axis=1), drop_first=True)
    y = (df['Attrition'] == 'Yes').astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop 10 Feature Importances:")
    print(importances.head(10))
    
    importances.plot(kind='bar', figsize=(10,6))
    plt.title('Feature Importance (Random Forest Baseline)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('experiments/eda/feature_importance.png')
    plt.show()

if __name__ == '__main__':
    eda()

