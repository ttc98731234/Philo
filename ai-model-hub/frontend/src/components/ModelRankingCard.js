import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Image } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';

const ModelRankingCard = ({ model, onPress }) => {
  // 假設model物件包含name, provider, score, icon, description等屬性
  
  return (
    <TouchableOpacity style={styles.container} onPress={onPress}>
      <LinearGradient
        colors={['#1e3c72', '#2a5298']}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.gradientBackground}
      >
        {/* 排名標記 */}
        <View style={styles.rankBadge}>
          <Text style={styles.rankText}>1</Text>
          <Ionicons name="trophy" size={14} color="#FFD700" style={styles.trophyIcon} />
        </View>
        
        {/* 模型信息 */}
        <View style={styles.contentContainer}>
          <View style={styles.modelIcon}>
            <Ionicons name={model?.icon || 'cube'} size={32} color="#FFF" />
          </View>
          
          <View style={styles.infoContainer}>
            <Text style={styles.modelName}>{model?.name || '最強AI模型'}</Text>
            <Text style={styles.providerName}>{model?.provider || '領先提供商'}</Text>
            
            {/* 模型評分 */}
            <View style={styles.scoreContainer}>
              <Text style={styles.scoreLabel}>綜合評分</Text>
              <View style={styles.scoreValueContainer}>
                <Text style={styles.scoreValue}>{model?.score || '9.8'}</Text>
                <View style={styles.stars}>
                  <Ionicons name="star" size={12} color="#FFD700" />
                  <Ionicons name="star" size={12} color="#FFD700" />
                  <Ionicons name="star" size={12} color="#FFD700" />
                  <Ionicons name="star" size={12} color="#FFD700" />
                  <Ionicons name="star-half" size={12} color="#FFD700" />
                </View>
              </View>
            </View>
          </View>
        </View>
        
        {/* 特點標籤 */}
        <View style={styles.tagsContainer}>
          {model?.tags?.map((tag, index) => (
            <View key={index} style={styles.tagBadge}>
              <Text style={styles.tagText}>{tag}</Text>
            </View>
          )) || (
            <>
              <View style={styles.tagBadge}>
                <Text style={styles.tagText}>精確回答</Text>
              </View>
              <View style={styles.tagBadge}>
                <Text style={styles.tagText}>知識豐富</Text>
              </View>
              <View style={styles.tagBadge}>
                <Text style={styles.tagText}>創意思考</Text>
              </View>
            </>
          )}
        </View>
        
        {/* 使用按鈕 */}
        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.useButton}>
            <Text style={styles.useButtonText}>使用此模型</Text>
            <Ionicons name="arrow-forward" size={16} color="#FFF" />
          </TouchableOpacity>
        </View>
        
        {/* 背景裝飾 */}
        <View style={styles.decorator} />
      </LinearGradient>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    margin: 15,
    borderRadius: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 5,
    height: 180,
  },
  gradientBackground: {
    borderRadius: 15,
    padding: 15,
    overflow: 'hidden',
    height: '100%',
  },
  rankBadge: {
    position: 'absolute',
    top: 10,
    right: 10,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    borderRadius: 12,
    paddingHorizontal: 10,
    paddingVertical: 4,
    flexDirection: 'row',
    alignItems: 'center',
  },
  rankText: {
    color: '#FFD700',
    fontWeight: 'bold',
    fontSize: 14,
    marginRight: 4,
  },
  trophyIcon: {
    marginLeft: 2,
  },
  contentContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 15,
  },
  modelIcon: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 15,
  },
  infoContainer: {
    flex: 1,
  },
  modelName: {
    color: '#FFF',
    fontSize: 22,
    fontWeight: 'bold',
    marginBottom: 2,
  },
  providerName: {
    color: 'rgba(255, 255, 255, 0.7)',
    fontSize: 14,
    marginBottom: 8,
  },
  scoreContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  scoreLabel: {
    color: 'rgba(255, 255, 255, 0.9)',
    fontSize: 12,
  },
  scoreValueContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  scoreValue: {
    color: '#FFF',
    fontWeight: 'bold',
    fontSize: 16,
    marginRight: 5,
  },
  stars: {
    flexDirection: 'row',
  },
  tagsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: 12,
  },
  tagBadge: {
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    borderRadius: 10,
    paddingHorizontal: 8,
    paddingVertical: 4,
    marginRight: 6,
    marginBottom: 6,
  },
  tagText: {
    color: '#FFF',
    fontSize: 10,
    fontWeight: '500',
  },
  buttonContainer: {
    alignItems: 'flex-start',
  },
  useButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.25)',
    borderRadius: 20,
    paddingHorizontal: 15,
    paddingVertical: 8,
    flexDirection: 'row',
    alignItems: 'center',
  },
  useButtonText: {
    color: '#FFF',
    fontSize: 14,
    fontWeight: '600',
    marginRight: 5,
  },
  decorator: {
    position: 'absolute',
    width: 150,
    height: 150,
    borderRadius: 75,
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    bottom: -50,
    right: -30,
  }
});

export default ModelRankingCard; 