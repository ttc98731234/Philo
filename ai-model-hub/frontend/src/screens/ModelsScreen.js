import React, { useState } from 'react';
import { StyleSheet, View, Text, FlatList, TouchableOpacity, TextInput } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';

const ModelsScreen = ({ navigation }) => {
  // 搜索查詢狀態
  const [searchQuery, setSearchQuery] = useState('');
  
  // 模型分類標籤
  const categories = [
    { id: '1', name: '全部' },
    { id: '2', name: 'OpenAI' },
    { id: '3', name: 'Anthropic' },
    { id: '4', name: 'Google' },
    { id: '5', name: 'Meta' },
    { id: '6', name: '其他' },
  ];
  
  // 活動標籤狀態
  const [activeCategory, setActiveCategory] = useState('1');
  
  // 模型數據
  const models = [
    { 
      id: '1', 
      name: 'GPT-4', 
      provider: 'OpenAI', 
      description: '最先進的大型語言模型，能夠理解和生成自然語言或代碼。',
      capabilities: ['文本理解', '代碼生成', '邏輯推理', '創意寫作'],
      rating: 4.9,
    },
    { 
      id: '2', 
      name: 'Claude 3', 
      provider: 'Anthropic', 
      description: '先進的AI助手，專注於有用、無害和誠實的回答。',
      capabilities: ['文本理解', '問題解答', '內容生成', '安全對齊'],
      rating: 4.8,
    },
    { 
      id: '3', 
      name: 'Gemini', 
      provider: 'Google', 
      description: '多模態模型，支持文本、圖像、音頻和視頻理解。',
      capabilities: ['多模態理解', '代碼生成', '數學推理', '知識檢索'],
      rating: 4.7,
    },
    { 
      id: '4', 
      name: 'Llama 3', 
      provider: 'Meta', 
      description: '開源大型語言模型，提供強大的自然語言處理能力。',
      capabilities: ['文本理解', '內容生成', '開源部署', '可自定義'],
      rating: 4.6,
    },
    { 
      id: '5', 
      name: 'Mixtral', 
      provider: 'Mistral AI', 
      description: '混合專家模型，提供平衡的性能和效率。',
      capabilities: ['文本生成', '專家混合', '多語言支持', '高效推理'],
      rating: 4.5,
    },
  ];
  
  // 篩選模型
  const filteredModels = models.filter(model => {
    // 按搜索查詢篩選
    const matchesSearch = model.name.toLowerCase().includes(searchQuery.toLowerCase()) || 
                        model.provider.toLowerCase().includes(searchQuery.toLowerCase());
    
    // 按類別篩選
    const matchesCategory = activeCategory === '1' || 
                          (activeCategory === '2' && model.provider === 'OpenAI') ||
                          (activeCategory === '3' && model.provider === 'Anthropic') ||
                          (activeCategory === '4' && model.provider === 'Google') ||
                          (activeCategory === '5' && model.provider === 'Meta') ||
                          (activeCategory === '6' && !['OpenAI', 'Anthropic', 'Google', 'Meta'].includes(model.provider));
    
    return matchesSearch && matchesCategory;
  });
  
  // 渲染模型項目
  const renderModelItem = ({ item }) => (
    <TouchableOpacity style={styles.modelCard}>
      <View style={styles.modelHeader}>
        <View>
          <Text style={styles.modelName}>{item.name}</Text>
          <Text style={styles.modelProvider}>{item.provider}</Text>
        </View>
        <View style={styles.ratingContainer}>
          <Ionicons name="star" size={16} color="#FFD700" />
          <Text style={styles.ratingText}>{item.rating}</Text>
        </View>
      </View>
      
      <Text style={styles.modelDescription}>{item.description}</Text>
      
      <View style={styles.capabilitiesContainer}>
        {item.capabilities.map((capability, index) => (
          <View key={index} style={styles.capabilityTag}>
            <Text style={styles.capabilityText}>{capability}</Text>
          </View>
        ))}
      </View>
      
      <View style={styles.actionButtonsContainer}>
        <TouchableOpacity style={styles.actionButton}>
          <Text style={styles.actionButtonText}>使用</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.actionButton, styles.outlineButton]}>
          <Text style={styles.outlineButtonText}>詳情</Text>
        </TouchableOpacity>
      </View>
    </TouchableOpacity>
  );
  
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>AI 模型</Text>
      </View>
      
      {/* 搜索欄 */}
      <View style={styles.searchContainer}>
        <Ionicons name="search" size={20} color="#666" style={styles.searchIcon} />
        <TextInput
          style={styles.searchInput}
          placeholder="搜索模型..."
          value={searchQuery}
          onChangeText={setSearchQuery}
        />
        {searchQuery.length > 0 && (
          <TouchableOpacity onPress={() => setSearchQuery('')} style={styles.clearButton}>
            <Ionicons name="close-circle" size={20} color="#999" />
          </TouchableOpacity>
        )}
      </View>
      
      {/* 分類標籤 */}
      <View style={styles.categoriesContainer}>
        <FlatList
          data={categories}
          horizontal
          showsHorizontalScrollIndicator={false}
          keyExtractor={(item) => item.id}
          renderItem={({ item }) => (
            <TouchableOpacity
              style={[
                styles.categoryTab,
                activeCategory === item.id && styles.activeCategoryTab
              ]}
              onPress={() => setActiveCategory(item.id)}
            >
              <Text
                style={[
                  styles.categoryText,
                  activeCategory === item.id && styles.activeCategoryText
                ]}
              >
                {item.name}
              </Text>
            </TouchableOpacity>
          )}
          contentContainerStyle={styles.categoriesList}
        />
      </View>
      
      {/* 模型列表 */}
      <FlatList
        data={filteredModels}
        keyExtractor={(item) => item.id}
        renderItem={renderModelItem}
        contentContainerStyle={styles.modelsList}
        showsVerticalScrollIndicator={false}
        ListEmptyComponent={() => (
          <View style={styles.emptyState}>
            <Ionicons name="search-outline" size={50} color="#999" />
            <Text style={styles.emptyStateText}>沒有找到符合條件的模型</Text>
          </View>
        )}
      />
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
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    marginHorizontal: 20,
    marginTop: 10,
    marginBottom: 15,
    borderRadius: 10,
    padding: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  searchIcon: {
    marginRight: 10,
  },
  searchInput: {
    flex: 1,
    fontSize: 16,
    color: '#333',
  },
  clearButton: {
    padding: 5,
  },
  categoriesContainer: {
    marginBottom: 15,
  },
  categoriesList: {
    paddingHorizontal: 15,
  },
  categoryTab: {
    paddingVertical: 8,
    paddingHorizontal: 16,
    marginHorizontal: 5,
    borderRadius: 20,
    backgroundColor: '#fff',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 1,
  },
  activeCategoryTab: {
    backgroundColor: '#4a6ee0',
  },
  categoryText: {
    fontSize: 14,
    color: '#666',
  },
  activeCategoryText: {
    color: '#fff',
    fontWeight: '500',
  },
  modelsList: {
    padding: 15,
  },
  modelCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 15,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  modelHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 10,
  },
  modelName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  modelProvider: {
    fontSize: 14,
    color: '#666',
    marginTop: 2,
  },
  ratingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f8f8f8',
    borderRadius: 12,
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  ratingText: {
    fontSize: 14,
    fontWeight: '500',
    color: '#333',
    marginLeft: 4,
  },
  modelDescription: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
    marginBottom: 12,
  },
  capabilitiesContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: 15,
  },
  capabilityTag: {
    backgroundColor: '#f0f4ff',
    borderRadius: 12,
    paddingHorizontal: 10,
    paddingVertical: 5,
    marginRight: 8,
    marginBottom: 8,
  },
  capabilityText: {
    fontSize: 12,
    color: '#4a6ee0',
  },
  actionButtonsContainer: {
    flexDirection: 'row',
  },
  actionButton: {
    flex: 1,
    backgroundColor: '#4a6ee0',
    borderRadius: 8,
    paddingVertical: 10,
    alignItems: 'center',
    marginRight: 10,
  },
  actionButtonText: {
    color: '#fff',
    fontWeight: '500',
    fontSize: 14,
  },
  outlineButton: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: '#4a6ee0',
  },
  outlineButtonText: {
    color: '#4a6ee0',
    fontWeight: '500',
    fontSize: 14,
  },
  emptyState: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 50,
  },
  emptyStateText: {
    fontSize: 16,
    color: '#999',
    marginTop: 10,
  },
});

export default ModelsScreen; 