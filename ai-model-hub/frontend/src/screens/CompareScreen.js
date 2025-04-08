import React, { useState } from 'react';
import { StyleSheet, View, Text, TouchableOpacity, TextInput, ScrollView, Switch } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';

const CompareScreen = ({ navigation }) => {
  // 狀態管理
  const [selectedModels, setSelectedModels] = useState([]);
  const [prompt, setPrompt] = useState('');
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(1000);
  const [isComparing, setIsComparing] = useState(false);
  const [compareResults, setCompareResults] = useState(null);
  
  // 可用模型列表
  const availableModels = [
    { id: '1', name: 'GPT-4', provider: 'OpenAI' },
    { id: '2', name: 'Claude 3', provider: 'Anthropic' },
    { id: '3', name: 'Gemini', provider: 'Google' },
    { id: '4', name: 'Llama 3', provider: 'Meta' },
    { id: '5', name: 'Mixtral', provider: 'Mistral AI' },
  ];
  
  // 切換模型選擇
  const toggleModelSelection = (modelId) => {
    if (selectedModels.includes(modelId)) {
      setSelectedModels(selectedModels.filter(id => id !== modelId));
    } else {
      if (selectedModels.length < 3) {
        setSelectedModels([...selectedModels, modelId]);
      }
    }
  };
  
  // 提交比較請求
  const submitComparison = () => {
    if (selectedModels.length < 2) {
      // 在真實應用中，應該顯示錯誤提示
      return;
    }
    
    if (!prompt.trim()) {
      // 在真實應用中，應該顯示錯誤提示
      return;
    }
    
    // 模擬 API 調用
    setIsComparing(true);
    
    // 模擬延遲
    setTimeout(() => {
      // 模擬回應數據
      const results = selectedModels.map(modelId => {
        const model = availableModels.find(m => m.id === modelId);
        return {
          modelId,
          modelName: model.name,
          provider: model.provider,
          response: `這是來自 ${model.name} 的回應示例。在實際應用中，這將是模型根據提示生成的文本。這只是用於演示目的的佔位符文本。`,
          responseTime: Math.floor(Math.random() * 2000) + 500, // 模擬響應時間 (ms)
          tokenCount: Math.floor(Math.random() * 500) + 100, // 模擬令牌計數
        };
      });
      
      setCompareResults(results);
      setIsComparing(false);
    }, 2000);
  };
  
  // 渲染模型選擇器
  const renderModelSelector = () => (
    <View style={styles.modelSelectorContainer}>
      <Text style={styles.sectionTitle}>選擇模型進行比較 (最多3個)</Text>
      <View style={styles.modelList}>
        {availableModels.map(model => (
          <TouchableOpacity
            key={model.id}
            style={[
              styles.modelChip,
              selectedModels.includes(model.id) && styles.selectedModelChip
            ]}
            onPress={() => toggleModelSelection(model.id)}
          >
            <Text
              style={[
                styles.modelChipText,
                selectedModels.includes(model.id) && styles.selectedModelChipText
              ]}
            >
              {model.name}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );
  
  // 渲染提示輸入
  const renderPromptInput = () => (
    <View style={styles.promptInputContainer}>
      <Text style={styles.sectionTitle}>輸入提示</Text>
      <TextInput
        style={styles.promptTextInput}
        multiline
        numberOfLines={4}
        placeholder="輸入你想要讓模型回答的問題或提示..."
        value={prompt}
        onChangeText={setPrompt}
      />
    </View>
  );
  
  // 渲染進階設置
  const renderAdvancedSettings = () => (
    <View style={styles.advancedSettingsContainer}>
      <TouchableOpacity
        style={styles.advancedSettingsToggle}
        onPress={() => setShowAdvancedSettings(!showAdvancedSettings)}
      >
        <Text style={styles.advancedSettingsToggleText}>進階設置</Text>
        <Ionicons
          name={showAdvancedSettings ? 'chevron-up' : 'chevron-down'}
          size={20}
          color="#666"
        />
      </TouchableOpacity>
      
      {showAdvancedSettings && (
        <View style={styles.advancedSettingsContent}>
          <View style={styles.settingRow}>
            <Text style={styles.settingLabel}>溫度</Text>
            <View style={styles.settingSliderContainer}>
              <Text style={styles.settingValue}>{temperature.toFixed(1)}</Text>
            </View>
          </View>
          
          <View style={styles.settingRow}>
            <Text style={styles.settingLabel}>最大令牌數</Text>
            <View style={styles.settingSliderContainer}>
              <Text style={styles.settingValue}>{maxTokens}</Text>
            </View>
          </View>
          
          <View style={styles.settingRow}>
            <Text style={styles.settingLabel}>使用緩存</Text>
            <Switch value={true} />
          </View>
        </View>
      )}
    </View>
  );
  
  // 渲染比較結果
  const renderCompareResults = () => (
    <View style={styles.resultsContainer}>
      <Text style={styles.sectionTitle}>比較結果</Text>
      
      {compareResults.map((result, index) => (
        <View key={result.modelId} style={styles.resultCard}>
          <View style={styles.resultHeader}>
            <View>
              <Text style={styles.resultModelName}>{result.modelName}</Text>
              <Text style={styles.resultModelProvider}>{result.provider}</Text>
            </View>
            <View style={styles.resultMetadataContainer}>
              <Text style={styles.resultMetadata}>{result.responseTime}ms</Text>
              <Text style={styles.resultMetadata}>{result.tokenCount} 令牌</Text>
            </View>
          </View>
          
          <Text style={styles.resultResponse}>{result.response}</Text>
          
          <View style={styles.resultActions}>
            <TouchableOpacity style={styles.resultActionButton}>
              <Ionicons name="copy-outline" size={16} color="#4a6ee0" />
              <Text style={styles.resultActionText}>複製</Text>
            </TouchableOpacity>
            
            <TouchableOpacity style={styles.resultActionButton}>
              <Ionicons name="thumbs-up-outline" size={16} color="#4a6ee0" />
              <Text style={styles.resultActionText}>好評</Text>
            </TouchableOpacity>
            
            <TouchableOpacity style={styles.resultActionButton}>
              <Ionicons name="thumbs-down-outline" size={16} color="#4a6ee0" />
              <Text style={styles.resultActionText}>差評</Text>
            </TouchableOpacity>
          </View>
        </View>
      ))}
      
      <TouchableOpacity style={styles.newComparisonButton} onPress={() => setCompareResults(null)}>
        <Text style={styles.newComparisonButtonText}>新的比較</Text>
      </TouchableOpacity>
    </View>
  );
  
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>模型比較</Text>
      </View>
      
      <ScrollView contentContainerStyle={styles.scrollContent}>
        {!compareResults ? (
          // 比較表單
          <>
            {renderModelSelector()}
            {renderPromptInput()}
            {renderAdvancedSettings()}
            
            <TouchableOpacity
              style={[
                styles.compareButton,
                (selectedModels.length < 2 || !prompt.trim()) && styles.disabledCompareButton
              ]}
              onPress={submitComparison}
              disabled={selectedModels.length < 2 || !prompt.trim() || isComparing}
            >
              {isComparing ? (
                <Text style={styles.compareButtonText}>比較中...</Text>
              ) : (
                <Text style={styles.compareButtonText}>開始比較</Text>
              )}
            </TouchableOpacity>
          </>
        ) : (
          // 比較結果
          renderCompareResults()
        )}
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f7',
  },
  header: {
    paddingHorizontal: 20,
    paddingVertical: 10,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
  },
  scrollContent: {
    padding: 20,
  },
  modelSelectorContainer: {
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 12,
  },
  modelList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  modelChip: {
    backgroundColor: '#fff',
    borderRadius: 20,
    paddingHorizontal: 15,
    paddingVertical: 8,
    marginRight: 10,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  selectedModelChip: {
    backgroundColor: '#4a6ee0',
    borderColor: '#4a6ee0',
  },
  modelChipText: {
    fontSize: 14,
    color: '#666',
  },
  selectedModelChipText: {
    color: '#fff',
  },
  promptInputContainer: {
    marginBottom: 20,
  },
  promptTextInput: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 15,
    height: 120,
    textAlignVertical: 'top',
    fontSize: 16,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  advancedSettingsContainer: {
    marginBottom: 20,
  },
  advancedSettingsToggle: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 10,
  },
  advancedSettingsToggleText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  advancedSettingsContent: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 15,
    marginTop: 10,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  settingRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 10,
  },
  settingLabel: {
    fontSize: 14,
    color: '#666',
  },
  settingSliderContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  settingValue: {
    fontSize: 14,
    fontWeight: '500',
    color: '#333',
    marginLeft: 10,
  },
  compareButton: {
    backgroundColor: '#4a6ee0',
    borderRadius: 12,
    paddingVertical: 15,
    alignItems: 'center',
    marginTop: 10,
  },
  disabledCompareButton: {
    backgroundColor: '#b0bec5',
  },
  compareButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  resultsContainer: {
    marginBottom: 30,
  },
  resultCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 15,
    marginBottom: 15,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  resultHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 10,
  },
  resultModelName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  resultModelProvider: {
    fontSize: 12,
    color: '#666',
  },
  resultMetadataContainer: {
    alignItems: 'flex-end',
  },
  resultMetadata: {
    fontSize: 12,
    color: '#999',
  },
  resultResponse: {
    fontSize: 14,
    color: '#333',
    lineHeight: 20,
    marginBottom: 15,
  },
  resultActions: {
    flexDirection: 'row',
    borderTopWidth: 1,
    borderTopColor: '#eee',
    paddingTop: 10,
  },
  resultActionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: 20,
  },
  resultActionText: {
    fontSize: 14,
    color: '#4a6ee0',
    marginLeft: 5,
  },
  newComparisonButton: {
    backgroundColor: '#fff',
    borderRadius: 12,
    paddingVertical: 15,
    alignItems: 'center',
    marginTop: 10,
    borderWidth: 1,
    borderColor: '#4a6ee0',
  },
  newComparisonButtonText: {
    color: '#4a6ee0',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default CompareScreen; 