def preprocess(data):
    # Frequency Encoding for 'category', 'city', and 'state'
    if 'category' in data.columns:
        data['category'] = data['category'].map(data['category'].value_counts(normalize=True))
    if 'city' in data.columns:
        data['city'] = data['city'].map(data['city'].value_counts(normalize=True))
    if 'state' in data.columns:
        data['state'] = data['state'].map(data['state'].value_counts(normalize=True))

    # Binary Encoding for 'gender'
    if 'gender' in data.columns:
        data['gender'] = data['gender'].map({'F': 0, 'M': 1})

    # Transform 'dob' into 'age' if 'dob' column exists
    if 'dob' in data.columns:
        data['dob'] = pd.to_datetime(data['dob'], errors='coerce')
        current_year = pd.to_datetime('today').year
        data['age'] = current_year - data['dob'].dt.year
        data = data.drop('dob', axis=1)

    # Ensure 'is_fraud' column is binary and integer
    data['is_fraud'] = data['is_fraud'].astype(int)

    # ---------------- FIX STARTS HERE ----------------
    # Separate target from features
    X = data.drop(columns=['is_fraud'])
    y = data['is_fraud']

    # Scale numerical features
    numerical_columns = X.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    # Encode categorical features
    categorical_columns = X.select_dtypes(include=['object']).columns
    encoder = ce.TargetEncoder(cols=categorical_columns)
    X_encoded = encoder.fit_transform(X, y)
    # ---------------- FIX ENDS HERE ----------------

    # Handle missing/infinite values
    if X_encoded.isna().sum().sum() > 0:
        X_encoded = X_encoded.fillna(0)
    if np.isinf(X_encoded).sum().sum() > 0:
        X_encoded = X_encoded.replace([np.inf, -np.inf], 0)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, encoder, scaler
