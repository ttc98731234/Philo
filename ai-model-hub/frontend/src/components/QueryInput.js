import React, { useState } from 'react';
import { View, TextInput, TouchableOpacity, StyleSheet, Keyboard, Animated } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

const QueryInput = ({ onSubmit, placeholder = '請輸入您的問題...' }) => {
  const [query, setQuery] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const animatedWidth = useState(new Animated.Value(0))[0];

  const handleSubmit = () => {
    if (query.trim()) {
      onSubmit(query.trim());
      setQuery('');
      Keyboard.dismiss();
    }
  };

  const handleFocus = () => {
    setIsFocused(true);
    Animated.timing(animatedWidth, {
      toValue: 1,
      duration: 300,
      useNativeDriver: false,
    }).start();
  };

  const handleBlur = () => {
    setIsFocused(false);
    if (!query.trim()) {
      Animated.timing(animatedWidth, {
        toValue: 0,
        duration: 300,
        useNativeDriver: false,
      }).start();
    }
  };

  const borderWidth = animatedWidth.interpolate({
    inputRange: [0, 1],
    outputRange: [1, 2],
  });

  return (
    <View style={styles.container}>
      <Animated.View
        style={[
          styles.inputContainer,
          {
            borderWidth,
            borderColor: isFocused ? '#4a6ee0' : '#ccc',
            backgroundColor: isFocused ? '#f8f9ff' : '#f0f0f0',
          },
        ]}
      >
        <Ionicons
          name="search"
          size={20}
          color={isFocused ? '#4a6ee0' : '#666'}
          style={styles.searchIcon}
        />
        <TextInput
          style={styles.input}
          value={query}
          onChangeText={setQuery}
          placeholder={placeholder}
          placeholderTextColor="#999"
          onFocus={handleFocus}
          onBlur={handleBlur}
          onSubmitEditing={handleSubmit}
          returnKeyType="search"
        />
        {query.length > 0 && (
          <TouchableOpacity
            style={styles.clearButton}
            onPress={() => setQuery('')}
          >
            <Ionicons name="close-circle" size={18} color="#999" />
          </TouchableOpacity>
        )}
      </Animated.View>

      <TouchableOpacity
        style={[
          styles.submitButton,
          { opacity: query.trim() ? 1 : 0.6 },
        ]}
        onPress={handleSubmit}
        disabled={!query.trim()}
      >
        <Ionicons name="paper-plane" size={22} color="#fff" />
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    paddingHorizontal: 15,
    paddingVertical: 10,
    alignItems: 'center',
  },
  inputContainer: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: 25,
    paddingHorizontal: 15,
    height: 50,
  },
  searchIcon: {
    marginRight: 8,
  },
  input: {
    flex: 1,
    height: 50,
    fontSize: 16,
    color: '#333',
    paddingVertical: 8,
  },
  clearButton: {
    padding: 5,
  },
  submitButton: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: '#4a6ee0',
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: 10,
    shadowColor: '#4a6ee0',
    shadowOffset: {
      width: 0,
      height: 3,
    },
    shadowOpacity: 0.3,
    shadowRadius: 5,
    elevation: 5,
  },
});

export default QueryInput; 