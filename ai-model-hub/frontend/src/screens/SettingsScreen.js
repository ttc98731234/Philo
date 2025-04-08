import React, { useState } from 'react';
import { StyleSheet, View, Text, TouchableOpacity, Switch, ScrollView, Alert } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';

const SettingsScreen = ({ navigation }) => {
  // 設置狀態
  const [darkMode, setDarkMode] = useState(false);
  const [cacheEnabled, setCacheEnabled] = useState(true);
  const [analyticsEnabled, setAnalyticsEnabled] = useState(true);
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const [saveHistory, setSaveHistory] = useState(true);
  const [language, setLanguage] = useState('zh-TW');
  
  // 設置部分
  const settingsSections = [
    {
      title: '一般設置',
      settings: [
        {
          id: 'darkMode',
          icon: 'moon',
          title: '深色模式',
          type: 'switch',
          value: darkMode,
          onToggle: setDarkMode,
        },
        {
          id: 'language',
          icon: 'language',
          title: '語言',
          type: 'select',
          value: language === 'zh-TW' ? '繁體中文' : (language === 'zh-CN' ? '簡體中文' : 'English'),
          onPress: () => showLanguageSelector(),
        },
        {
          id: 'notifications',
          icon: 'notifications',
          title: '通知',
          type: 'switch',
          value: notificationsEnabled,
          onToggle: setNotificationsEnabled,
        },
      ],
    },
    {
      title: '數據與存儲',
      settings: [
        {
          id: 'cache',
          icon: 'save',
          title: '啟用快取',
          description: '使用緩存來節省token並提高響應速度',
          type: 'switch',
          value: cacheEnabled,
          onToggle: setCacheEnabled,
        },
        {
          id: 'history',
          icon: 'time',
          title: '保存歷史記錄',
          type: 'switch',
          value: saveHistory,
          onToggle: setSaveHistory,
        },
        {
          id: 'clearCache',
          icon: 'trash',
          title: '清除緩存',
          description: '刪除所有緩存的回應',
          type: 'button',
          onPress: () => confirmClearCache(),
        },
      ],
    },
    {
      title: '隱私與安全',
      settings: [
        {
          id: 'analytics',
          icon: 'analytics',
          title: '匿名使用數據',
          description: '幫助我們改進應用',
          type: 'switch',
          value: analyticsEnabled,
          onToggle: setAnalyticsEnabled,
        },
        {
          id: 'deleteData',
          icon: 'warning',
          title: '刪除所有數據',
          description: '永久刪除您的所有數據和偏好設置',
          type: 'button',
          danger: true,
          onPress: () => confirmDeleteData(),
        },
      ],
    },
    {
      title: '關於',
      settings: [
        {
          id: 'version',
          icon: 'information-circle',
          title: '版本',
          description: '1.0.0',
          type: 'info',
        },
        {
          id: 'terms',
          icon: 'document-text',
          title: '使用條款',
          type: 'navigation',
          onPress: () => {},
        },
        {
          id: 'privacy',
          icon: 'shield-checkmark',
          title: '隱私政策',
          type: 'navigation',
          onPress: () => {},
        },
      ],
    },
  ];
  
  // 語言選擇器
  const showLanguageSelector = () => {
    // 在實際應用中，這裡會顯示一個底部表單或模態窗口
    Alert.alert(
      '選擇語言',
      '',
      [
        { text: '繁體中文', onPress: () => setLanguage('zh-TW') },
        { text: '簡體中文', onPress: () => setLanguage('zh-CN') },
        { text: 'English', onPress: () => setLanguage('en') },
        { text: '取消', style: 'cancel' },
      ]
    );
  };
  
  // 確認清除緩存
  const confirmClearCache = () => {
    Alert.alert(
      '清除緩存',
      '您確定要清除所有緩存的回應嗎？',
      [
        { text: '取消', style: 'cancel' },
        { text: '清除', style: 'destructive', onPress: () => {} },
      ]
    );
  };
  
  // 確認刪除數據
  const confirmDeleteData = () => {
    Alert.alert(
      '刪除所有數據',
      '此操作無法撤銷。您確定要永久刪除所有數據嗎？',
      [
        { text: '取消', style: 'cancel' },
        { text: '刪除', style: 'destructive', onPress: () => {} },
      ]
    );
  };
  
  // 渲染設置項
  const renderSettingItem = (setting) => {
    return (
      <TouchableOpacity
        key={setting.id}
        style={[
          styles.settingItem,
          setting.type === 'button' && setting.danger && styles.dangerSettingItem
        ]}
        onPress={setting.type === 'switch' ? null : setting.onPress}
        disabled={setting.type === 'info'}
      >
        <View style={styles.settingIconContainer}>
          <Ionicons
            name={setting.icon}
            size={22}
            color={setting.type === 'button' && setting.danger ? '#e53935' : '#4a6ee0'}
          />
        </View>
        
        <View style={styles.settingTextContainer}>
          <Text
            style={[
              styles.settingTitle,
              setting.type === 'button' && setting.danger && styles.dangerSettingTitle
            ]}
          >
            {setting.title}
          </Text>
          {setting.description && (
            <Text style={styles.settingDescription}>{setting.description}</Text>
          )}
        </View>
        
        {setting.type === 'switch' && (
          <Switch
            value={setting.value}
            onValueChange={setting.onToggle}
            trackColor={{ false: '#d1d1d6', true: '#b0c3e2' }}
            thumbColor={setting.value ? '#4a6ee0' : '#f4f3f4'}
          />
        )}
        
        {setting.type === 'select' && (
          <View style={styles.settingSelectContainer}>
            <Text style={styles.settingSelectValue}>{setting.value}</Text>
            <Ionicons name="chevron-forward" size={16} color="#999" />
          </View>
        )}
        
        {setting.type === 'navigation' && (
          <Ionicons name="chevron-forward" size={16} color="#999" />
        )}
      </TouchableOpacity>
    );
  };
  
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>設置</Text>
      </View>
      
      <ScrollView>
        {settingsSections.map((section) => (
          <View key={section.title} style={styles.section}>
            <Text style={styles.sectionTitle}>{section.title}</Text>
            <View style={styles.sectionContent}>
              {section.settings.map(renderSettingItem)}
            </View>
          </View>
        ))}
        
        <TouchableOpacity style={styles.signOutButton}>
          <Text style={styles.signOutButtonText}>登出</Text>
        </TouchableOpacity>
        
        <Text style={styles.footerText}>AI Model Hub © 2024</Text>
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
  section: {
    marginBottom: 25,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    marginLeft: 20,
    marginBottom: 10,
  },
  sectionContent: {
    backgroundColor: '#fff',
    borderRadius: 12,
    marginHorizontal: 15,
    overflow: 'hidden',
  },
  settingItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 14,
    paddingHorizontal: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  dangerSettingItem: {
    backgroundColor: '#fff8f8',
  },
  settingIconContainer: {
    width: 36,
    height: 36,
    borderRadius: 8,
    backgroundColor: '#f0f4ff',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 15,
  },
  settingTextContainer: {
    flex: 1,
  },
  settingTitle: {
    fontSize: 16,
    fontWeight: '500',
    color: '#333',
  },
  dangerSettingTitle: {
    color: '#e53935',
  },
  settingDescription: {
    fontSize: 14,
    color: '#999',
    marginTop: 3,
  },
  settingSelectContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  settingSelectValue: {
    fontSize: 15,
    color: '#999',
    marginRight: 5,
  },
  signOutButton: {
    backgroundColor: '#fff',
    marginHorizontal: 15,
    marginTop: 10,
    marginBottom: 20,
    borderRadius: 12,
    padding: 15,
    alignItems: 'center',
  },
  signOutButtonText: {
    fontSize: 16,
    fontWeight: '500',
    color: '#e53935',
  },
  footerText: {
    textAlign: 'center',
    fontSize: 14,
    color: '#999',
    marginBottom: 30,
  },
});

export default SettingsScreen; 