import React from 'react';
import { StyleSheet, View, Text, ScrollView, TouchableOpacity, Image } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { SafeAreaView } from 'react-native-safe-area-context';

const HomeScreen = ({ navigation }) => {
  // 模型卡片數據
  const modelCards = [
    { id: '1', name: 'GPT-4', provider: 'OpenAI', icon: 'chatbubble-ellipses' },
    { id: '2', name: 'Claude 3', provider: 'Anthropic', icon: 'sparkles' },
    { id: '3', name: 'Gemini', provider: 'Google', icon: 'analytics' },
    { id: '4', name: 'Llama 3', provider: 'Meta', icon: 'code-working' },
  ];

  // 功能卡片數據
  const featureCards = [
    { id: '1', title: '模型比較', description: '比較不同模型的回應', icon: 'git-compare' },
    { id: '2', title: '模型辯論', description: '讓AI模型互相辯論', icon: 'chatbubbles' },
    { id: '3', title: '混合回應', description: '結合多個模型的回應', icon: 'git-merge' },
    { id: '4', title: '深度搜索', description: '搜索詳細資訊進行回答', icon: 'search' },
  ];

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>AI Model Hub</Text>
        <TouchableOpacity style={styles.profileButton}>
          <Ionicons name="person-circle" size={30} color="#4a6ee0" />
        </TouchableOpacity>
      </View>

      <ScrollView showsVerticalScrollIndicator={false}>
        {/* 歡迎區域 */}
        <View style={styles.welcomeSection}>
          <Text style={styles.welcomeTitle}>歡迎使用 AI Model Hub</Text>
          <Text style={styles.welcomeSubtitle}>多AI模型調用、比較、混合平台</Text>
        </View>

        {/* 熱門模型區域 */}
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>熱門模型</Text>
            <TouchableOpacity>
              <Text style={styles.seeAllText}>查看全部</Text>
            </TouchableOpacity>
          </View>

          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.cardsScroll}>
            {modelCards.map((model) => (
              <TouchableOpacity key={model.id} style={styles.modelCard}>
                <View style={styles.modelIconContainer}>
                  <Ionicons name={model.icon} size={24} color="#fff" />
                </View>
                <Text style={styles.modelName}>{model.name}</Text>
                <Text style={styles.modelProvider}>{model.provider}</Text>
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>

        {/* 功能區域 */}
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>平台功能</Text>
          </View>

          <View style={styles.featureGrid}>
            {featureCards.map((feature) => (
              <TouchableOpacity key={feature.id} style={styles.featureCard}>
                <View style={styles.featureIconContainer}>
                  <Ionicons name={feature.icon} size={24} color="#4a6ee0" />
                </View>
                <Text style={styles.featureTitle}>{feature.title}</Text>
                <Text style={styles.featureDescription}>{feature.description}</Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>
        
        {/* 最近活動區域 */}
        <View style={[styles.section, { marginBottom: 30 }]}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>最近活動</Text>
          </View>
          
          <View style={styles.activityCard}>
            <Text style={styles.emptyStateText}>沒有最近活動記錄</Text>
          </View>
        </View>
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
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 10,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
  },
  profileButton: {
    padding: 5,
  },
  welcomeSection: {
    paddingHorizontal: 20,
    paddingVertical: 25,
  },
  welcomeTitle: {
    fontSize: 26,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
  },
  welcomeSubtitle: {
    fontSize: 16,
    color: '#666',
  },
  section: {
    marginTop: 20,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    marginBottom: 15,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  seeAllText: {
    fontSize: 14,
    color: '#4a6ee0',
  },
  cardsScroll: {
    paddingLeft: 20,
  },
  modelCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 15,
    marginRight: 15,
    width: 140,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  modelIconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#4a6ee0',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 10,
  },
  modelName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 5,
  },
  modelProvider: {
    fontSize: 14,
    color: '#666',
  },
  featureGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    paddingHorizontal: 15,
  },
  featureCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 15,
    margin: 5,
    width: '47%',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  featureIconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#f0f4ff',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 10,
  },
  featureTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 5,
  },
  featureDescription: {
    fontSize: 12,
    color: '#666',
  },
  activityCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 20,
    marginHorizontal: 20,
    alignItems: 'center',
    justifyContent: 'center',
    height: 100,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  emptyStateText: {
    fontSize: 14,
    color: '#999',
  },
});

export default HomeScreen; 